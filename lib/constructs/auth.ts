import { CfnOutput, Duration, RemovalPolicy } from "aws-cdk-lib";
import { Mfa, UserPool, UserPoolClient, UserPoolOperation } from "aws-cdk-lib/aws-cognito";
import { Runtime } from "aws-cdk-lib/aws-lambda";
import { NodejsFunction } from "aws-cdk-lib/aws-lambda-nodejs";
import { Construct } from "constructs";
import * as path from "path";

export interface AuthProps {
  selfSignUpEnabled: boolean;
  allowedSignUpEmailDomains: string[];
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
      selfSignUpEnabled: props.selfSignUpEnabled,
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
