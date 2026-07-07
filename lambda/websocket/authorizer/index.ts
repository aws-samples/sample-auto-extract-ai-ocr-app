// WebSocket API $connect ルート専用の Lambda Authorizer
// クエリ文字列の idToken を Cognito JWT として検証する。
// 認証のみに責務を絞り、DynamoDB への書き込みは行わない
// （$connect の統合Lambda 側で行う）。
//
// 参照実装: https://github.com/aws-samples/websocket-api-cognito-auth-sample
import { APIGatewayRequestAuthorizerHandler } from "aws-lambda";
import { CognitoJwtVerifier } from "aws-jwt-verify";

const UserPoolId = process.env.USER_POOL_ID!;
const AppClientId = process.env.APP_CLIENT_ID!;

export const handler: APIGatewayRequestAuthorizerHandler = async (event) => {
  try {
    const verifier = CognitoJwtVerifier.create({
      userPoolId: UserPoolId,
      tokenUse: "id",
      clientId: AppClientId,
    });

    const encodedToken = event.queryStringParameters?.idToken;
    if (!encodedToken) {
      throw new Error("idToken query string parameter is missing");
    }

    const payload = await verifier.verify(encodedToken);
    return allowPolicy(event.methodArn, payload.sub);
  } catch (error: any) {
    console.log("Authorization failed:", error.message);
    return denyAllPolicy();
  }
};

function allowPolicy(methodArn: string, cognitoSub: string) {
  return {
    principalId: cognitoSub,
    policyDocument: {
      Version: "2012-10-17",
      Statement: [
        {
          Action: "execute-api:Invoke",
          Effect: "Allow",
          Resource: methodArn,
        } as const,
      ],
    },
    context: {
      // 後続の websocketHandler の event.requestContext.authorizer.cognitoSub で参照可能
      cognitoSub,
    },
  };
}

function denyAllPolicy() {
  return {
    principalId: "*",
    policyDocument: {
      Version: "2012-10-17",
      Statement: [
        {
          Action: "*",
          Effect: "Deny",
          Resource: "*",
        } as const,
      ],
    },
  };
}
