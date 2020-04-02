import unittest
from data import ByteCode

class TestByteCode(unittest.TestCase):

    def test_encode(self):
        ds = ["hello", '']
        bc = ByteCode("byte_values.txt")
        for i in range(len(ds)):
            indices = bc.to_int_seq(ds[i])
            self.assertEqual(indices[-1], bc.STOP_CODE)
            self.assertEqual(ds[i], bc.to_string(indices))


if __name__ == "__main__":
    unittest.main()