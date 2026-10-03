import argparse
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from server import Backend, validate


class ServerContract(unittest.TestCase):
    def test_rejects_invalid_generation_parameters(self):
        for override in ({"n": 0}, {"n": 5}, {"n": True}, {"steps": 0}, {"steps": 101},
                         {"seed": -1}, {"size": "1025x1024"}, {"size": "2048x2048"},
                         {"response_format": "url"}, {"model": "wrong"}, {"prompt": ""}):
            with self.subTest(override=override), self.assertRaises(ValueError):
                validate(dict(prompt="test", **{k:v for k,v in override.items() if k != "prompt"})
                         if "prompt" not in override else override)

    def test_multipart_numbers_and_auto_size(self):
        fields = validate(dict(prompt="test", n="4", seed="42", steps="25", size="auto"))
        self.assertEqual((fields["n"], fields["seed"], fields["width"], fields["height"]), (4, 42, 1024, 1024))

    def test_edit_references_reach_both_encoder_and_vae(self):
        backend = object.__new__(Backend)
        backend.args = argparse.Namespace(unet="model.gguf", clip="encoder.safetensors", precision="nvfp4")
        graph = backend.graph(validate(dict(prompt="turn it blue", seed=42)), ["a.png", "b.png"], "test")
        self.assertEqual(graph["5"]["inputs"]["vae"], ["3", 0])
        self.assertEqual(graph["5"]["inputs"]["images.image_1"], ["11", 0])
        self.assertEqual(graph["5"]["inputs"]["images.image_2"], ["12", 0])
        self.assertEqual(graph["7"]["inputs"]["cfg"], 1.)
        self.assertEqual(graph["7"]["inputs"]["scheduler"], "simple")


if __name__ == "__main__":
    unittest.main()
