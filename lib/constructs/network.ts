import { CfnOutput } from "aws-cdk-lib";
import { SubnetType, Vpc } from "aws-cdk-lib/aws-ec2";
import { Construct } from "constructs";

export class Network extends Construct {
  public readonly vpc: Vpc;
  constructor(scope: Construct, id: string) {
    super(scope, id);

    const vpc = new Vpc(this, "Vpc", {
      maxAzs: 2,
      subnetConfiguration: [
        {
          subnetType: SubnetType.PUBLIC,
          mapPublicIpOnLaunch: false,
          name: "Public",
        },
        {
          subnetType: SubnetType.PRIVATE_WITH_EGRESS,
          name: "Private",
        },
      ],
    });
    this.vpc = vpc;

    new CfnOutput(this, "VpcId", {
      value: vpc.vpcId,
      description: "VPC ID",
    });
  }
}
