// ImagesTable から image_id -> app_name を取得するヘルパー。
// lambda/api/app/repositories/image_repository.py の get_image と同義
// （app_name フィールドのみを参照する軽量版）。
import { DynamoDBClient } from "@aws-sdk/client-dynamodb";
import { DynamoDBDocumentClient, GetCommand } from "@aws-sdk/lib-dynamodb";

const ddbClient = DynamoDBDocumentClient.from(new DynamoDBClient({}));
const ImagesTableName = process.env.IMAGES_TABLE_NAME!;

export async function getAppNameByImageId(imageId: string): Promise<string | null> {
  const res = await ddbClient.send(
    new GetCommand({
      TableName: ImagesTableName,
      Key: { id: imageId },
      ProjectionExpression: "app_name",
    })
  );
  return (res.Item?.app_name as string) ?? null;
}
