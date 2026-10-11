"""Small checkpoint-default comparison, with independently checked chain commits."""
from time import perf_counter_ns

QUESTION_IDS = (401, 421, 441, 461)
CHAIN_METHODS = ('eagle_chain3_no_cache', 'eagle_chain3_cache')
CONFIG_SHA256 = 'f611623deb1803b4e208e97adf822f15f0fa23e81d11806cf00e18096104a4f3'


def verify_chain_cycles(row, reference):
    """Reconstruct maximal accepted prefixes from actual target argmax values."""
    offset = 0
    for i, (cycle, metric) in enumerate(zip(row['cycles'], row['cycle_metrics'], strict=True)):
        query, greedy = cycle['verification_tokens'], cycle['verifier_tokens']
        drafts = query[1:]
        expected_drafts = min(3, row['total_tokens'] - row['input_tokens'] - offset - 1)
        proof = row['query_proofs'][i+1]
        if (len(drafts) != expected_drafts or len(greedy) != len(query)
                or proof['logits_shape'][1] != len(query) or proof['greedy_tokens'] != greedy
                or cycle['selected_raw_ids'] != list(range(1, len(query)))
                or cycle['base_kv_length'] != row['input_tokens'] + offset
                or cycle['proposal_prefix_tokens'] != cycle['base_kv_length']):
            raise ValueError('native chain query or prefix differs')
        accepted = 0
        for proposed, predicted in zip(drafts, greedy):
            if proposed != predicted:
                break
            accepted += 1
        committed = query[:accepted+1]
        for j, token in enumerate(committed):
            if token in (151643, 151645):
                committed = committed[:j+1]
                break
        if (cycle['committed_tokens'] != committed
                or metric['accepted_draft_nodes'] != len(committed)-1
                or tuple(metric['accepted_raw_ids']) != tuple(range(1, len(committed)))
                or cycle['kv_length_after'] != cycle['base_kv_length'] + len(committed)
                or cycle['observed_kv_length_after'] != cycle['kv_length_after']):
            raise ValueError('native chain acceptance or KV crop differs')
        offset += len(committed)
    if reference is not None and row['cycles'] != reference['cycles']:
        raise ValueError('native chain queries or commits differ from reference')


class NativeChainRecorder:
    """Delegate proposals to the checkpoint; retain tensors until outside timing."""
    def __init__(self, drafter, cache_provider):
        self.drafter, self.cache_provider = drafter, cache_provider
        self.layer_indices = drafter.layer_indices
        self.rows, self.proposals = [], []
        self.offset = 0

    def propose(self, **kwargs):
        import torch
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        began = perf_counter_ns()
        result = self.drafter.propose(**kwargs)
        elapsed = perf_counter_ns() - began
        end.record()
        self.proposals.append((kwargs['fed_ids'].shape[1], elapsed, start, end))
        return result

    def begin_cycle(self, *, iteration, base_kv_length, verification_token_ids):
        self.rows.append(dict(iteration=iteration, committed_offset=self.offset,
            base_kv_length=base_kv_length, verification_tokens=verification_token_ids.detach()))

    def end_cycle(self, *, verifier_tokens, accepted_draft_tokens, committed_token_ids, kv_length_after):
        self.rows[-1].update(verifier_tokens=verifier_tokens.detach(),
            committed_tokens=committed_token_ids.detach(), kv_length_after=kv_length_after,
            observed_kv_length_after=self.cache_provider().get_seq_length())
        self.offset += committed_token_ids.numel()

    def abort_cycle(self):
        pass

    def finish_generation(self, **kwargs):
        pass

    def materialize(self):
        rows, metrics = [], []
        for source, (prefix, wall, start, end) in zip(self.rows, self.proposals, strict=True):
            row = {k: v.flatten().cpu().tolist() if hasattr(v, 'detach') else v for k, v in source.items()}
            n, a = len(row['verification_tokens'])-1, len(row['committed_tokens'])-1
            row.update(selected_raw_ids=list(range(1, n+1)), proposal_prefix_tokens=prefix)
            rows.append(row)
            metrics.append(dict(iteration=row['iteration'], committed_output_offset=row['committed_offset'],
                selected_draft_nodes=n, accepted_draft_nodes=a, verification_raw_ids=list(range(n+1)),
                accepted_raw_ids=list(range(1, a+1)), proposal_wall_ns=wall,
                proposal_cuda_ms=start.elapsed_time(end)))
        return rows, metrics
