import json
from pathlib import Path
import unittest

from .generate import ROOT
from .priority import order_cases


def case(op, form="fp32", mode="ii", width=8):
    return dict(op=op, form=form, mode=mode, width=width, name=f"{op}_{form}_{mode}_{width}")


class PriorityTests(unittest.TestCase):
    def test_common_and_exu2_before_other_operations(self):
        a, b, c = case("VCMP_EQ"), case("VADD"), case("VPACKV2")
        self.assertEqual(order_cases([a,c,b], exu2={("VPACKV2","fp32")}), [b,c,a])

    def test_basic_ii_breadth_before_forwarding_and_wide_variants(self):
        a, b, c = case("VADD",mode="forwarding",width=6), case("VAXPY"), case("VADD",width=16)
        self.assertEqual(order_cases([a,c,b],exu2=set()), [b,a,c])

    def test_common_forms_before_packed_forms(self):
        a, b = case("VADD","f8e4m3"), case("VAXPY","fp32")
        self.assertEqual(order_cases([a,b],exu2=set()), [b,a])

    def test_unsupported_inputs_last_even_for_exu2(self):
        a = dict(case("VPACKV2","f4e1m2x2"), input_dtype="uint32",semantic_input_dtype="f4e1m2x2")
        b = case("UNKNOWN","f8e4m3")
        self.assertEqual(order_cases([a,b],exu2={(a["op"],a["form"])}), [b,a])

    def test_manifest_escape_hatch_and_no_mutation(self):
        values = [case("VMUL"),case("VADD")]
        before = json.dumps(values)
        self.assertEqual(order_cases(values,"manifest"), values)
        self.assertNotEqual(order_cases(values,exu2=set()), values)
        self.assertEqual(json.dumps(values),before)

    def test_every_suite_keeps_all_cases_and_same_content(self):
        for directory in ("a6_instruction","a6_exu2","a6_a5_coverage","a6_calibration"):
            cases = json.loads((ROOT / "op_kernel" / directory / "manifest.json").read_text())["cases"]
            ordered = order_cases(cases)
            self.assertEqual(len(ordered),len(cases))
            self.assertEqual({c["name"]:c for c in cases},{c["name"]:c for c in ordered})
            self.assertEqual(ordered, order_cases(list(reversed(cases))))


if __name__ == "__main__":
    unittest.main()
