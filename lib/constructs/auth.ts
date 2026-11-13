import { CfnOutput, Duration, RemovalPolicy } from "aws-cdk-lib";
import { Mfa, UserPool, UserPoolClient, UserPoolOperation } from "aws-cdk-lib/aws-cognito";
import { Runtime } from "aws-cdk-lib/aws-lambda";
import { NodejsFunction } from "aws-cdk-lib/aws-lambda-nodejs";
import { PolicyStatement } from "aws-cdk-lib/aws-iam";
import { Construct } from "constructs";
import * as path from "path";

export interface AuthProps {
  selfSignUpEnabled: boolean;
  allowedSignUpEmailDomains: string[];
  samlAuthEnabled: boolean;
}

export class Auth extends Construct {
  readonly userPool: UserPool;
  readonly client: UserPoolClient;
  constructor(scope: Construct, id: string, props: AuthProps) {
    super(scope, id);

    const userPool = new UserPool(this, "UserPool", {
      removalPolicy: RemovalPolicy.DESTROY,
      passwordPolicy: {
        minLength: 8,
        requireLowercase: true,
        requireUppercase: true,
        requireDigits: true,
        requireSymbols: true,
      },
      // SAML ではログインできるユーザーを IdP 側で決めるため塞ぐ
      selfSignUpEnabled: props.samlAuthEnabled ? false : props.selfSignUpEnabled,
      signInAliases: {
        username: false,
        email: true,
        phone: false,
      },
    });

    if (props.allowedSignUpEmailDomains.length > 0) {
      const preSignUp = new NodejsFunction(this, "PreSignUpFunction", {
        runtime: Runtime.NODEJS_20_X,
        entry: path.join(__dirname, "../../lambda/pre-signup/index.ts"),
        handler: "handler",
        timeout: Duration.seconds(5),
        environment: {
          ALLOWED_DOMAINS: JSON.stringify(props.allowedSignUpEmailDomains),
        },
      });
      userPool.addTrigger(UserPoolOperation.PRE_SIGN_UP, preSignUp);
    }

    // DSQL へのユーザー同期は Post Authentication トリガーでは行わず、API の POST /user/sync で行う。
    // Cognito はトリガーの応答を 5 秒しか待たず、DSQL への接続とトランザクションが間に合わないとログイン自体が失敗するため。

    const client = userPool.addClient("UserPoolClient", {
      idTokenValidity: Duration.days(1),
    });

    new CfnOutput(this, "UserPoolId", {
      value: userPool.userPoolId,
    });

    new CfnOutput(this, "UserPoolClientId", {
      value: client.userPoolClientId,
    });

    this.client = client;
    this.userPool = userPool;
  }
}
