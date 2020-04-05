import unittest
import torch
from data import ByteCode, ByteDataset, ByteDataLoader

class TestByteCode(unittest.TestCase):

    def test_encode(self):
        ds = ["hello", '']
        bc = ByteCode("byte_values.txt")
        for i in range(len(ds)):
            indices = bc.to_int_seq(ds[i])
            self.assertEqual(indices[-1], bc.STOP_CODE)
            self.assertEqual(ds[i], bc.to_string(indices))

    def test_to_int_seq(self):
        s = ' '
        bc = ByteCode("byte_values.txt")
        self.assertEqual(bc.to_int_seq(s), [bc.STOP_CODE, bc._byte_value_map[32], bc.STOP_CODE])

    def test__to_code(self):
        bc = ByteCode("byte_values.txt")
        not_a_byteval = -3
        self.assertEqual(bc._to_code(not_a_byteval), bc._byte_value_map[bc.MISSING])

class TestDataLoader(unittest.TestCase):

    def test_dl_formats(self):
        bc = ByteCode("byte_values.txt")
        ds = ByteDataset("_bios.json", bc)
        dl = ByteDataLoader(ds, pack_onehot=False)
        onehot, targets = next(iter(dl))
        self.assertTrue(isinstance(onehot, list))

        dl = ByteDataLoader(ds, pack_onehot=True)
        onehot, targets = next(iter(dl))
    

if __name__ == "__main__":
    unittest.main()