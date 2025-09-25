import * as s3 from "aws-cdk-lib/aws-s3";
import { Construct } from "constructs";
import { aws_cloudfront, CfnOutput, RemovalPolicy, Stack } from "aws-cdk-lib";
import {
  CloudFrontToS3,
  CloudFrontToS3Props,
} from "@aws-solutions-constructs/aws-cloudfront-s3";
import { NodejsBuild } from "deploy-time-build";

export interface WebProps {
  buildFolder: string;
  userPoolId: string;
  userPoolClientId: string;
  apiUrl: string;
  enableOcr: boolean;
}
export class Web extends Construct {
  constructor(scope: Construct, id: string, props: WebProps) {
    super(scope, id);

    const { buildFolder, userPoolId, userPoolClientId, apiUrl, enableOcr } = props;

    const bucketProps: s3.BucketProps = {
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      encryption: s3.BucketEncryption.S3_MANAGED,
      autoDeleteObjects: true,
      removalPolicy: RemovalPolicy.DESTROY,
      objectOwnership: s3.ObjectOwnership.OBJECT_WRITER,
      enforceSSL: true,
      cors: [
        {
          allowedMethods: [
            s3.HttpMethods.GET,
            s3.HttpMethods.PUT,
            s3.HttpMethods.POST,
            s3.HttpMethods.HEAD,
          ],
          allowedOrigins: ["*"],
          allowedHeaders: ["*"],
          exposedHeaders: [
            "ETag",
            "x-amz-meta-custom-header",
            "x-amz-server-side-encryption",
            "x-amz-request-id",
            "x-amz-id-2",
          ],
        },
      ],
    };

    const cloudFrontToS3Props: CloudFrontToS3Props = {
      insertHttpSecurityHeaders: false,
      bucketProps: bucketProps,
      loggingBucketProps: bucketProps,
      cloudFrontLoggingBucketProps: bucketProps,
      cloudFrontLoggingBucketAccessLogBucketProps: bucketProps,
      cloudFrontDistributionProps: {
        minimumProtocolVersion:
          aws_cloudfront.SecurityPolicyProtocol.TLS_V1_2_2021,
        geoRestriction: aws_cloudfront.GeoRestriction.allowlist("JP"),
        errorResponses: [
          {
            httpStatus: 403,
            responseHttpStatus: 200,
            responsePagePath: "/index.html",
          },
          {
            httpStatus: 404,
            responseHttpStatus: 200,
            responsePagePath: "/index.html",
          },
        ],
        defaultRootObject: "index.html",
      },
    };

    const { cloudFrontWebDistribution, s3BucketInterface } = new CloudFrontToS3(
      this,
      "Web",
      cloudFrontToS3Props
    );

    new NodejsBuild(this, "WebBuild", {
      assets: [
        {
          path: "web",
          exclude: [
            "dist",
            "dev-dist",
            "node_modules",
            ".env",
            ".sample-env",
            "README.md",
            ".gitignore",
            "*.tsbuildinfo",
          ],
        },
      ],
      destinationBucket: s3BucketInterface,
      distribution: cloudFrontWebDistribution,
      outputSourceDirectory: buildFolder,
      buildCommands: ["npm i", "npm run build"],
      buildEnvironment: {
        VITE_APP_USER_POOL_ID: userPoolId,
        VITE_APP_USER_POOL_CLIENT_ID: userPoolClientId,
        VITE_APP_REGION: Stack.of(this).region,
        VITE_API_BASE_URL: apiUrl,
        VITE_ENABLE_OCR: enableOcr.toString(),
      },
    });

    new CfnOutput(this, "CloudFrontURL", {
      value: `https://${cloudFrontWebDistribution.domainName}`,
    });
  }
}
