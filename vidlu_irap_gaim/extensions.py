from collections.abc import Mapping
from vidlu.training.extensions import TrainerExtension


class MultiAttributeScorePrinter(TrainerExtension):
    """
    Prints metrics to stdout after evaluation completes, with special formatting
    for per-attribute metrics (nested dictionaries).
    """

    def initialize(self, trainer):
        @trainer.evaluation.completed.handler
        def on_eval_completed(state):
            if not hasattr(state, "metrics"):
                return

            print("\nEvaluation Results:")

            # Separate scalar metrics from per-attribute metrics
            scalars = {}
            per_attr_metrics = {}

            for name, value in state.metrics.items():
                if isinstance(value, Mapping):
                    per_attr_metrics[name] = value
                else:
                    try:
                        val = float(value)
                        scalars[name] = val
                    except Exception:
                        pass  # Skip non-scalar/non-dict stuff

            # Print scalars first (Compact One-Lines)
            if scalars:
                scalar_str = ", ".join(f"{k}: {v:.4f}" for k, v in sorted(scalars.items()))
                print(f"Aggregate Metrics: {scalar_str}")

            if not per_attr_metrics:
                print("\n")
                return

            print("\nPer-Attribute Metrics:")

            # Format as Table
            # 1. Collect all attribute keys union
            all_attrs = set()
            for m_dict in per_attr_metrics.values():
                all_attrs.update(m_dict.keys())

            # Sort attributes
            try:
                # Try sorting by string representation for consistency
                sorted_attrs = sorted(all_attrs, key=lambda x: str(x))
            except Exception:
                sorted_attrs = list(all_attrs)

            # Columns: Attribute, Metric1, Metric2...
            # Metric headers: Display name (lstrip _)
            metric_names = sorted(per_attr_metrics.keys(), key=lambda x: x.lstrip("_"))
            headers = ["Attribute"] + [m.lstrip("_") for m in metric_names]

            # Determine column widths
            col_widths = [len(h) for h in headers]

            # Helper to get formatted value
            def get_val_str(m_name, attr):
                if m_name not in per_attr_metrics or attr not in per_attr_metrics[m_name]:
                    return "-"
                val = per_attr_metrics[m_name][attr]
                try:
                    fval = float(val)
                    return f"{fval:.4f}"
                except Exception:
                    return str(val)

            # Pre-calculate rows to determine widths
            rows = []
            for attr in sorted_attrs:
                row = [str(attr)]
                col_widths[0] = max(col_widths[0], len(str(attr)))

                for i, m_name in enumerate(metric_names):
                    s = get_val_str(m_name, attr)
                    row.append(s)
                    col_widths[i + 1] = max(col_widths[i + 1], len(s))
                rows.append(row)

            # Print Header
            # Attribute (string) left align. Numbers right align.
            header_strs = []
            header_strs.append(headers[0].ljust(col_widths[0]))
            for i in range(1, len(headers)):
                header_strs.append(headers[i].rjust(col_widths[i]))

            print("  " + "  ".join(header_strs))
            print("  " + "  ".join(["-" * w for w in col_widths]))

            # Print Rows
            for row in rows:
                row_strs = []
                row_strs.append(row[0].ljust(col_widths[0]))
                for i in range(1, len(row)):
                    row_strs.append(row[i].rjust(col_widths[i]))
                print("  " + "  ".join(row_strs))

            print("\n")
