import { PreSignUpTriggerEvent, PreSignUpTriggerHandler } from "aws-lambda";

export const handler: PreSignUpTriggerHandler = async (
  event: PreSignUpTriggerEvent
) => {
  const allowedDomains: string[] = JSON.parse(
    process.env.ALLOWED_DOMAINS || "[]"
  );
  if (allowedDomains.length === 0) return event;

  const email = event.request.userAttributes.email || "";
  const domain = email.split("@")[1]?.toLowerCase();

  if (!domain || !allowedDomains.includes(domain)) {
    throw new Error(`Email domain "${domain}" is not allowed.`);
  }

  return event;
};
