"""CPU summaries of actual verification and accepted paths in serial records."""
from collections import Counter


def acceptance_lengths(rows):
    """Count tree nodes, accepted draft nodes, and actual committed paths.

    The root is included in verification and committed lengths, but excluded
    from accepted draft lengths. A final pending token without a target forward
    is not a verification round. AR rounds have length one and no draft nodes.
    """
    verification, accepted, committed = [], [], []
    selected_total = 0
    for row in rows:
        forwards = row["scheduler"]["forwards"]
        if row["method"] == "target_only":
            if row["cycle_metrics"] or row["cycles"] or forwards != row["output_tokens"]-1:
                raise ValueError("AR verification coverage differs")
            verification.extend([1]*forwards)
            accepted.extend([0]*forwards)
            committed.extend([1]*forwards)
            continue
        metrics, cycles = row["cycle_metrics"], row["cycles"]
        if len(metrics) != len(cycles) or len(metrics) != forwards:
            raise ValueError("tree verification coverage differs")
        offset = 0
        for index, (metric, cycle) in enumerate(zip(metrics, cycles, strict=True)):
            selected = cycle["selected_raw_ids"]
            tokens = cycle["committed_tokens"]
            n, a = metric["selected_draft_nodes"], metric["accepted_draft_nodes"]
            if (type(n) is not int or type(a) is not int or not 0 <= a <= n
                    or metric["iteration"] != index or cycle["iteration"] != index
                    or metric["committed_output_offset"] != offset or cycle["committed_offset"] != offset
                    or tuple(metric["verification_raw_ids"]) != (0, *selected)
                    or n != len(selected) or len(set(selected)) != n
                    or len(tokens) != a+1 or len(metric["accepted_raw_ids"]) != a
                    or any(raw not in selected for raw in metric["accepted_raw_ids"])
                    or tokens != row["output_token_ids"][offset:offset+len(tokens)]):
                raise ValueError("verification or accepted path differs from actual commits")
            verification.append(n+1)
            accepted.append(a)
            committed.append(len(tokens))
            selected_total += n
            offset += len(tokens)
        if not 0 <= row["output_tokens"]-offset <= 1:
            raise ValueError("unaccounted generation outside verification rounds")

    def summary(values):
        return dict(total=sum(values), mean=sum(values)/len(values) if values else None,
                    histogram={str(k): v for k, v in sorted(Counter(values).items())})

    return dict(verification_rounds=len(verification),
                verification_length_including_root=summary(verification),
                accepted_draft_length=summary(accepted),
                committed_length_including_root=summary(committed),
                selected_draft_nodes=selected_total,
                accepted_fraction_of_selected_nodes=sum(accepted)/selected_total if selected_total else None)
