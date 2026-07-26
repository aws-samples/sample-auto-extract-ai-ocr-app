import { CfnOutput, Duration, RemovalPolicy } from "aws-cdk-lib";
import { Mfa, UserPool, UserPoolClient, UserPoolOperation } from "aws-cdk-lib/aws-cognito";
import { DockerImageCode, DockerImageFunction, Runtime } from "aws-cdk-lib/aws-lambda";
import { NodejsFunction } from "aws-cdk-lib/aws-lambda-nodejs";
import { PolicyStatement } from "aws-cdk-lib/aws-iam";
import { Platform } from "aws-cdk-lib/aws-ecr-assets";
import { Construct } from "constructs";
import * as path from "path";

export interface AuthProps {
  selfSignUpEnabled: boolean;
  allowedSignUpEmailDomains: string[];
}

export class Auth extends Construct {
  readonly userPool: UserPool;
  readonly client: UserPoolClient;
  readonly postAuthFunction: DockerImageFunction;
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

    // Post Authentication Trigger（Cognito → DSQL 同期）
    const postAuth = new DockerImageFunction(this, "PostAuthFunction", {
      code: DockerImageCode.fromImageAsset("lambda/post-auth", {
        platform: Platform.LINUX_AMD64,
      }),
      timeout: Duration.seconds(30),
    });
    userPool.addTrigger(UserPoolOperation.POST_AUTHENTICATION, postAuth);
    this.postAuthFunction = postAuth;

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
