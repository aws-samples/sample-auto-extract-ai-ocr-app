import { CfnOutput, Duration, RemovalPolicy } from "aws-cdk-lib";
import { Mfa, UserPool, UserPoolClient } from "aws-cdk-lib/aws-cognito";
import { Construct } from "constructs";

export class Auth extends Construct {
  readonly userPool: UserPool;
  readonly client: UserPoolClient;
  constructor(scope: Construct, id: string) {
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
      selfSignUpEnabled: true,
      signInAliases: {
        username: false,
        email: true,
        phone: false,
      },
    });

    const client = userPool.addClient("UserPoolClient", {
      idTokenValidity: Duration.days(1),
    });

    new CfnOutput(this, "UserPoolId", {
      value: userPool.userPoolId,
      description: "UserPool ID",
    });

    new CfnOutput(this, "UserPoolClientId", {
      value: client.userPoolClientId,
      description: "UserPoolClientId",
    });

    this.client = client;
    this.userPool = userPool;
  }
}
