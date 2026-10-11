import gzip
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import prepare_evaluation_workloads as workloads


class EvaluationWorkloadsTest(unittest.TestCase):
    def source(self, name, rows):
        data="".join(json.dumps(row)+"\n" for row in rows).encode()
        if name == "humaneval":
            data=gzip.compress(data,mtime=0)
        return data,dict(bytes=len(data),sha256=sha256(data).hexdigest(),count=len(rows))

    def test_humaneval_hidden_tests_and_reference_are_not_in_prompt(self):
        row=dict(task_id="HumanEval/0",prompt="def f():\n    pass\n",entry_point="f",
                 test="hidden_test_marker",canonical_solution="hidden_solution_marker")
        data,spec=self.source("humaneval",[row])
        actual=workloads.normalize("humaneval",data,spec)[0]
        self.assertNotIn("hidden_",actual["prompt"])
        self.assertIn(row["prompt"],actual["prompt"])
        self.assertEqual(actual["evaluation"]["test"],row["test"])
        self.assertEqual(actual["source_id"],"HumanEval/0")

    def test_aime_integer_and_padded_string_answers_are_normalized_separately(self):
        for answer in (70,"070"):
            data,spec=self.source("aime25",[dict(id="0",problem="A problem?",answer=answer)])
            actual=workloads.normalize("aime25",data,spec)[0]
            self.assertEqual(actual["evaluation"]["answer"],"70")
            self.assertNotIn("70",actual["prompt"])
        for answer in (True,-1,1000,"2.5"):
            data,spec=self.source("aime25",[dict(id="0",problem="A problem?",answer=answer)])
            with self.assertRaises(ValueError):workloads.normalize("aime25",data,spec)

    def test_summarization_prompt_is_verbatim_and_reference_stays_in_evaluation(self):
        rows=[dict(question_id=241+i,category="summarization",prompt=f"Summarize: article {i}",
                   reference=[f"reference_marker {i}"]) for i in range(2)]
        data,spec=self.source("summarization",rows)
        spec["source_question_ids"]=[241,242]
        actual=workloads.normalize("summarization",data,spec)
        self.assertEqual([r["question_id"] for r in actual],[0,1])
        self.assertEqual([r["source_id"] for r in actual],["241","242"])
        self.assertEqual(actual[1]["prompt"],"Summarize: article 1")
        self.assertEqual(actual[1]["evaluation"],dict(kind="reference_summary_not_scored",reference="reference_marker 1"))
        self.assertNotIn("reference_marker",actual[1]["prompt"])
        for change in (dict(question_id=243),dict(question_id=242.0),dict(category="qa"),dict(prompt="article 1"),
                       dict(reference=["a","b"]),dict(reference=[""])):
            data,spec=self.source("summarization",[rows[0],dict(rows[1],**change)])
            spec["source_question_ids"]=[241,242]
            with self.assertRaises(ValueError):workloads.normalize("summarization",data,spec)

    def test_summarization_lock_pins_the_whole_specbench_subset(self):
        spec=json.loads(workloads.LOCK.read_text())["datasets"]["summarization"]
        first,last=spec["source_question_ids"]
        self.assertEqual((first,last-first+1,spec["count"]),(241,80,80))
        self.assertIn(spec["revision"],spec["url"])
        self.assertTrue(spec["url"].endswith("/"+spec["filename"]))

    def test_rejects_changed_bytes_count_and_source_order(self):
        rows=[dict(id="0",problem="A problem?",answer=7)]
        data,spec=self.source("aime25",rows)
        with self.assertRaises(ValueError):workloads.normalize("aime25",data+b" ",spec)
        with self.assertRaises(ValueError):workloads.normalize("aime25",data,dict(spec,count=2))
        data,spec=self.source("aime25",rows*2)
        with self.assertRaises(ValueError):workloads.normalize("aime25",data,spec)

    def test_preparation_is_reusable_and_refuses_overwriting_modified_data(self):
        data,spec=self.source("aime25",[dict(id="0",problem="A problem?",answer=7)])
        spec.update(revision="fixture",filename="test.jsonl")
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/"source";source.mkdir()
            (source/"test.jsonl").write_bytes(data)
            lock=root/"lock.json"
            lock.write_text(json.dumps(dict(protocol="fixture",datasets=dict(aime25=spec))))
            with patch.object(workloads,"LOCK",lock):
                path,receipt=workloads.prepare("aime25",root/"out",source_directory=source)
                self.assertEqual(receipt["status"],"prepared_not_measured")
                self.assertEqual(workloads.prepare("aime25",root/"out"),(path,receipt))
                path.write_bytes(b"user modification")
                with self.assertRaises(ValueError):workloads.prepare("aime25",root/"out")
                self.assertEqual(path.read_bytes(),b"user modification")


if __name__ == "__main__":
    unittest.main()
