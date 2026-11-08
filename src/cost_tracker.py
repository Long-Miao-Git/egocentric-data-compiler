"""成本追踪模块"""

from .config import Config
from .schemas import CostRecord


class CostTracker:
    """记录各模块token消耗和费用"""

    def __init__(self):
        self.records: list[CostRecord] = []
        rates = Config.cost_rates()
        self._vlm_in = rates.get("vlm_input_per_1k", 0.02)
        self._vlm_out = rates.get("vlm_output_per_1k", 0.06)
        self._llm_in = rates.get("llm_input_per_1k", 0.004)
        self._llm_out = rates.get("llm_output_per_1k", 0.012)

    def record_vlm(self, input_tokens: int, output_tokens: int, module: str = "vlm"):
        cost = (input_tokens / 1000 * self._vlm_in) + (output_tokens / 1000 * self._vlm_out)
        self.records.append(CostRecord(
            module=module,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_cny=round(cost, 4),
        ))

    def record_llm(self, input_tokens: int, output_tokens: int, module: str = "llm"):
        cost = (input_tokens / 1000 * self._llm_in) + (output_tokens / 1000 * self._llm_out)
        self.records.append(CostRecord(
            module=module,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_cny=round(cost, 4),
        ))

    def total_cost(self) -> float:
        return round(sum(r.estimated_cost_cny for r in self.records), 4)

    def total_tokens(self) -> tuple[int, int]:
        total_in = sum(r.input_tokens for r in self.records)
        total_out = sum(r.output_tokens for r in self.records)
        return (total_in, total_out)

    def get_report(self) -> str:
        lines = ["=== 成本报告 ==="]
        for r in self.records:
            lines.append(f"  {r.module}: in={r.input_tokens} out={r.output_tokens} cost=¥{r.estimated_cost_cny}")
        total_in, total_out = self.total_tokens()
        lines.append(f"---")
        lines.append(f"  总计: in={total_in} out={total_out} cost=¥{self.total_cost()}")
        return "\n".join(lines)
