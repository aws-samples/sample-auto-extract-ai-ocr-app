"""AgentCore Gateway Lambda Target: calc-verify"""

import json
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def verify_unit_price_calculation(quantity: float, unit_price: float, expected_amount: float) -> dict:
    """単価計算の検算"""
    calculated_amount = quantity * unit_price
    is_correct = abs(calculated_amount - expected_amount) < 0.01

    result = {
        "is_correct": is_correct,
        "calculated_amount": calculated_amount,
        "expected_amount": expected_amount,
        "quantity": quantity,
        "unit_price": unit_price,
    }

    if is_correct:
        result["message"] = "単価計算は正しいです"
    else:
        result["message"] = f"単価計算が間違っています。{quantity} × {unit_price} = {calculated_amount}"

    return result


def verify_subtotal_calculation(amounts: list[float], expected_subtotal: float) -> dict:
    """小計計算の検算"""
    calculated_subtotal = sum(amounts)
    is_correct = abs(calculated_subtotal - expected_subtotal) < 0.01

    result = {
        "is_correct": is_correct,
        "calculated_subtotal": calculated_subtotal,
        "expected_subtotal": expected_subtotal,
        "amounts": amounts,
    }

    if is_correct:
        result["message"] = "小計の計算は正しいです"
    else:
        result["message"] = f"小計が間違っています。{' + '.join(map(str, amounts))} = {calculated_subtotal}"

    return result


def verify_total_with_tax_calculation(subtotal: float, tax_amount: float, expected_total: float) -> dict:
    """税込み合計計算の検算"""
    calculated_total = subtotal + tax_amount
    is_correct = abs(calculated_total - expected_total) < 0.01

    result = {
        "is_correct": is_correct,
        "calculated_total": calculated_total,
        "expected_total": expected_total,
        "subtotal": subtotal,
        "tax_amount": tax_amount,
    }

    if is_correct:
        result["message"] = "税込み合計の計算は正しいです"
    else:
        result["message"] = f"税込み合計が間違っています。{subtotal} + {tax_amount} = {calculated_total}"

    return result


def verify_tax_calculation(subtotal: float, tax_rate: float, actual_tax_amount: float) -> dict:
    """消費税計算の検算"""
    calculated_tax = subtotal * tax_rate
    is_correct = abs(calculated_tax - actual_tax_amount) < 0.01

    result = {
        "is_correct": is_correct,
        "calculated_tax": calculated_tax,
        "actual_tax_amount": actual_tax_amount,
        "subtotal": subtotal,
        "tax_rate": tax_rate,
    }

    if is_correct:
        if calculated_tax != actual_tax_amount:
            result["message"] = f"消費税の計算は正しいです（端数処理済み: 理論値{calculated_tax}円 → {actual_tax_amount}円）"
        else:
            result["message"] = "消費税の計算は正しいです"
    else:
        result["message"] = f"消費税が間違っています。{subtotal} × {tax_rate} = {calculated_tax}"

    return result


TOOL_HANDLERS = {
    "verify_unit_price_calculation": verify_unit_price_calculation,
    "verify_subtotal_calculation": verify_subtotal_calculation,
    "verify_total_with_tax_calculation": verify_total_with_tax_calculation,
    "verify_tax_calculation": verify_tax_calculation,
}


def _get_tool_name(context) -> str:
    """Extract tool name from Lambda context (set by AgentCore Gateway)."""
    try:
        client_context = context.client_context
        if client_context and hasattr(client_context, "custom"):
            custom = client_context.custom or {}
            tool_name = custom.get("bedrockAgentCoreToolName", "")
            # Strip gateway target prefix (e.g., "calc-verify___verify_subtotal_calculation")
            if "___" in tool_name:
                return tool_name.split("___", 1)[1]
            return tool_name
    except Exception as e:
        logger.warning(f"Failed to extract tool name from context: {e}")
    return ""


def handler(event, context):
    """Lambda handler for AgentCore Gateway Target."""
    logger.info(f"Received event: {json.dumps(event)}")

    tool_name = _get_tool_name(context)
    arguments = event
    logger.info(f"Tool name: {tool_name}")

    try:
        if tool_name in TOOL_HANDLERS:
            result = TOOL_HANDLERS[tool_name](**arguments)
        else:
            return {"error": f"Unknown tool: {tool_name}"}

        return result
    except Exception as e:
        logger.error(f"Error: {e}")
        return {"error": str(e)}
