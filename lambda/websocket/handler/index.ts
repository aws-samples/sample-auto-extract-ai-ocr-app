// API Gateway WebSocket API の $connect / $disconnect / $default 統合Lambda。
//
// $connect: 認証（Lambda Authorizer）のみ通し、まだ resource_id には紐付けない。
// $default: クライアントからの action で分岐する。
//   - action: "subscribe"    → resource_id（image_id）への視聴登録。権限チェックを行う
//   - action: "subscribeAll" → 一覧ページ用の全体購読登録（resource_id="all"）。
//                               個々のimage_idへのsubscribe/unsubscribeイベントを受け取る
//   - action: "heartbeat"     → removed_at(TTL) を延長
//   - それ以外 → 400 を返す
// $disconnect: GSI(connection_id逆引き)でレコードを検索し削除（best-effort）。
//
// 配信（プレゼンス通知）は AppSync のような自動配信ではなく、このLambda自身が
// 「同じ resource_id を見ている他 connection」+「全体購読者(resource_id="all")」を
// 検索し PostToConnectionCommand で直接送信する自前実装。
//
// 参照実装: https://github.com/aws-samples/websocket-api-cognito-auth-sample
import { APIGatewayProxyHandler } from "aws-lambda";
import { DynamoDBClient } from "@aws-sdk/client-dynamodb";
import {
  DynamoDBDocumentClient,
  PutCommand,
  DeleteCommand,
  QueryCommand,
  ScanCommand,
} from "@aws-sdk/lib-dynamodb";
import {
  ApiGatewayManagementApiClient,
  PostToConnectionCommand,
} from "@aws-sdk/client-apigatewaymanagementapi";
import { canViewUsecase } from "./dsql";
import { getAppNameByImageId } from "./images";

const ddbClient = DynamoDBDocumentClient.from(new DynamoDBClient({}));
const ConnectionsTableName = process.env.CONNECTIONS_TABLE_NAME!;

// 全体購読（一覧ページ用）を表す特別な resource_id。
// 個別の image_id（"image#<id>"）とは名前空間が重複しない固定値。
const ALL_RESOURCE_ID = "all";

// Heartbeat: 5分間隔想定。TTLはその3倍の15分でバッファを持たせる。
// API Gateway WebSocket API はアイドル10分でハード切断されるため、
// Heartbeat間隔はこれより十分短くする必要がある（AWS公式の制約、変更不可）。
const TTL_SECONDS = 15 * 60;

function nowPlusTtl(): number {
  return Math.floor(Date.now() / 1000) + TTL_SECONDS;
}

function resourceIdFor(imageId: string): string {
  return `image#${imageId}`;
}

export const handler: APIGatewayProxyHandler = async (event) => {
  const routeKey = event.requestContext.routeKey!;
  const connectionId = event.requestContext.connectionId!;

  if (routeKey === "$connect") {
    // 認証は Authorizer 側で完了済み。resource_id はまだ不明なため、
    // このタイミングでは DynamoDB への書き込みは行わない（subscribe 時に行う）。
    return { statusCode: 200, body: "Connected." };
  }

  if (routeKey === "$disconnect") {
    const removedResourceIds = await removeConnectionByConnectionId(connectionId);
    const removedImageResourceIds = removedResourceIds.filter((rid) => rid.startsWith("image#"));
    // 個別ページ（同じ image_id を見ている他ユーザー）に、視聴解除後の最新の視聴者一覧を配信する。
    // これを行わないと、閉じた後もそのユーザーのバッジが他ユーザーの画面に残り続けてしまう。
    await Promise.all(removedImageResourceIds.map((rid) => broadcastPresence(event, rid)));
    // 一覧側（全体購読者）にも変化を通知する。
    if (removedImageResourceIds.length > 0) {
      await broadcastAllSummary(event);
    }
    return { statusCode: 200, body: "Disconnected." };
  }

  // $default ルート: action で分岐
  let body: any;
  try {
    body = JSON.parse(event.body ?? "{}");
  } catch {
    return { statusCode: 400, body: "Invalid JSON body." };
  }

  const cognitoSub: string | undefined = event.requestContext.authorizer?.cognitoSub;
  if (!cognitoSub) {
    return { statusCode: 401, body: "Unauthorized." };
  }

  switch (body.action) {
    case "subscribe":
      return handleSubscribe(event, connectionId, cognitoSub, body.imageId);
    case "subscribeAll":
      return handleSubscribeAll(event, connectionId, cognitoSub);
    case "heartbeat":
      return handleHeartbeat(connectionId, body.imageId);
    default:
      return { statusCode: 400, body: `Unknown action: ${body.action}` };
  }
};

async function handleSubscribe(
  event: Parameters<APIGatewayProxyHandler>[0],
  connectionId: string,
  cognitoSub: string,
  imageId: string | undefined
) {
  if (!imageId) {
    return { statusCode: 400, body: "imageId is required for subscribe." };
  }

  const appName = await getAppNameByImageId(imageId);
  if (!appName) {
    return { statusCode: 404, body: "Image not found." };
  }

  const { allowed, displayName } = await canViewUsecase(cognitoSub, appName);
  if (!allowed) {
    return { statusCode: 403, body: "Forbidden." };
  }

  const resourceId = resourceIdFor(imageId);

  await ddbClient.send(
    new PutCommand({
      TableName: ConnectionsTableName,
      Item: {
        resource_id: resourceId,
        connection_id: connectionId,
        user_id: cognitoSub,
        display_name: displayName,
        removed_at: nowPlusTtl(),
      },
    })
  );

  // 個別image_idの視聴者一覧を該当image_idの購読者へ配信
  await broadcastPresence(event, resourceId);
  // 全体購読者（一覧ページ）へも「このimage_idの視聴者が変わった」ことを通知
  await broadcastAllSummary(event);
  return { statusCode: 200, body: "Subscribed." };
}

/** 一覧ページ用: 全体購読の登録。特定のimage_idには紐付けない。 */
async function handleSubscribeAll(
  event: Parameters<APIGatewayProxyHandler>[0],
  connectionId: string,
  cognitoSub: string
) {
  await ddbClient.send(
    new PutCommand({
      TableName: ConnectionsTableName,
      Item: {
        resource_id: ALL_RESOURCE_ID,
        connection_id: connectionId,
        user_id: cognitoSub,
        removed_at: nowPlusTtl(),
      },
    })
  );

  // 登録直後に現在の全体状況を送る（他の配信を待たずに初期表示ができるように）
  await sendAllSummaryTo(event, connectionId);
  return { statusCode: 200, body: "SubscribedAll." };
}

async function handleHeartbeat(connectionId: string, imageId: string | undefined) {
  // imageId 未指定の場合は全体購読(ALL_RESOURCE_ID)のheartbeatとみなす
  const resourceId = imageId ? resourceIdFor(imageId) : ALL_RESOURCE_ID;

  // heartbeat は既存レコードの removed_at 延長のみ。user_id 等は subscribe 時点の値を維持するため
  // 全属性を伴う PutCommand ではなく、既存アイテムを Get せずに Put で上書きしても user_id は不明になるため、
  // ここでは Query して該当アイテムを取得し、その user_id を維持したまま Put する。
  const existing = await ddbClient.send(
    new QueryCommand({
      TableName: ConnectionsTableName,
      KeyConditionExpression: "resource_id = :rid AND connection_id = :cid",
      ExpressionAttributeValues: { ":rid": resourceId, ":cid": connectionId },
    })
  );
  const item = existing.Items?.[0];
  if (!item) {
    return { statusCode: 404, body: "Subscription not found. Please subscribe again." };
  }

  await ddbClient.send(
    new PutCommand({
      TableName: ConnectionsTableName,
      Item: { ...item, removed_at: nowPlusTtl() },
    })
  );
  return { statusCode: 200, body: "Heartbeat received." };
}

/**
 * GSI(connection_id逆引き)で該当レコードを検索し削除する（$disconnect用、best-effort）。
 * 削除された resource_id の一覧を返す（呼び出し元が全体購読者への通知判断に使う）。
 */
async function removeConnectionByConnectionId(connectionId: string): Promise<string[]> {
  const res = await ddbClient.send(
    new QueryCommand({
      TableName: ConnectionsTableName,
      IndexName: "ConnectionIdIndex",
      KeyConditionExpression: "connection_id = :cid",
      ExpressionAttributeValues: { ":cid": connectionId },
    })
  );
  const removedResourceIds: string[] = [];
  for (const item of res.Items ?? []) {
    await ddbClient.send(
      new DeleteCommand({
        TableName: ConnectionsTableName,
        Key: { resource_id: item.resource_id, connection_id: item.connection_id },
      })
    );
    removedResourceIds.push(item.resource_id);
  }
  return removedResourceIds;
}

function getManagementApiEndpoint(event: Parameters<APIGatewayProxyHandler>[0]): string {
  const domainName = event.requestContext.domainName!;
  return domainName.endsWith("amazonaws.com")
    ? `https://${domainName}/${event.requestContext.stage}`
    : `https://${domainName}`;
}

/** 指定 connection_id 群に payload を送信し、410 Gone のものは削除する */
async function postToConnections(
  managementApi: ApiGatewayManagementApiClient,
  items: Record<string, any>[],
  payload: Buffer
): Promise<void> {
  await Promise.all(
    items.map(async (item) => {
      try {
        await managementApi.send(
          new PostToConnectionCommand({
            ConnectionId: item.connection_id,
            Data: payload,
          })
        );
      } catch (e: any) {
        if (e.statusCode === 410 || e.$metadata?.httpStatusCode === 410) {
          // stale connection: 削除してクリーンアップ
          await ddbClient.send(
            new DeleteCommand({
              TableName: ConnectionsTableName,
              Key: { resource_id: item.resource_id, connection_id: item.connection_id },
            })
          );
        } else {
          console.error("Failed to post to connection", item.connection_id, e);
        }
      }
    })
  );
}

/** resource_id を見ている全 connection に、現在の視聴者一覧をブロードキャストする */
async function broadcastPresence(
  event: Parameters<APIGatewayProxyHandler>[0],
  resourceId: string
): Promise<void> {
  const res = await ddbClient.send(
    new QueryCommand({
      TableName: ConnectionsTableName,
      KeyConditionExpression: "resource_id = :rid",
      ExpressionAttributeValues: { ":rid": resourceId },
    })
  );
  const viewers = (res.Items ?? []).map((item) => ({
    userId: item.user_id,
    displayName: item.display_name ?? null,
  }));

  const managementApi = new ApiGatewayManagementApiClient({
    endpoint: getManagementApiEndpoint(event),
  });
  const payload = Buffer.from(
    JSON.stringify({ type: "presence", resourceId, viewers }),
    "utf-8"
  );

  await postToConnections(managementApi, res.Items ?? [], payload);
}

/**
 * 全体購読者（一覧ページ）向けに、現在「image#」が付く resource_id ごとの視聴者数マップを
 * 配信する。ConnectionsTable 全体を Scan するのではなく、GSI や別テーブルを使わず
 * シンプルにテーブル全体を Scan する実装（接続数が数十〜数百件規模の前提のため許容）。
 */
async function buildAllSummary(): Promise<Record<string, { userId: string; displayName: string | null }[]>> {
  const summary: Record<string, { userId: string; displayName: string | null }[]> = {};
  let lastEvaluatedKey: Record<string, any> | undefined;

  do {
    const res: any = await ddbClient.send(
      new ScanCommand({
        TableName: ConnectionsTableName,
        ExclusiveStartKey: lastEvaluatedKey,
      })
    );
    for (const item of res.Items ?? []) {
      const rid = item.resource_id as string;
      if (!rid.startsWith("image#")) continue;
      const imageId = rid.slice("image#".length);
      if (!summary[imageId]) summary[imageId] = [];
      summary[imageId].push({ userId: item.user_id, displayName: item.display_name ?? null });
    }
    lastEvaluatedKey = res.LastEvaluatedKey;
  } while (lastEvaluatedKey);

  return summary;
}

/** 全体購読者（resource_id = "all"）全員に現在の視聴状況マップを配信する */
async function broadcastAllSummary(event: Parameters<APIGatewayProxyHandler>[0]): Promise<void> {
  const [allSubscribers, byImageId] = await Promise.all([
    ddbClient.send(
      new QueryCommand({
        TableName: ConnectionsTableName,
        KeyConditionExpression: "resource_id = :rid",
        ExpressionAttributeValues: { ":rid": ALL_RESOURCE_ID },
      })
    ),
    buildAllSummary(),
  ]);

  const items = allSubscribers.Items ?? [];
  if (items.length === 0) return;

  const managementApi = new ApiGatewayManagementApiClient({
    endpoint: getManagementApiEndpoint(event),
  });
  const payload = Buffer.from(
    JSON.stringify({ type: "presence_all", byImageId }),
    "utf-8"
  );

  await postToConnections(managementApi, items, payload);
}

/** subscribeAll 登録直後、当該 connection にのみ現在の全体状況を送る */
async function sendAllSummaryTo(
  event: Parameters<APIGatewayProxyHandler>[0],
  connectionId: string
): Promise<void> {
  const byImageId = await buildAllSummary();
  const managementApi = new ApiGatewayManagementApiClient({
    endpoint: getManagementApiEndpoint(event),
  });
  const payload = Buffer.from(
    JSON.stringify({ type: "presence_all", byImageId }),
    "utf-8"
  );
  await postToConnections(managementApi, [{ resource_id: ALL_RESOURCE_ID, connection_id: connectionId }], payload);
}
