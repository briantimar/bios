import unittest
from data import ByteWrapper

class TestByteDataset(unittest.TestCase):

    def test_encode(self):
        ds = ["hello", '']
        bds = ByteWrapper(ds)
        for i in range(len(ds)):
            indices = bds[i]
            self.assertEqual(indices[-1], ByteWrapper.STOP_CODE)
            self.assertEqual(ds[i], bds._to_string(indices))


if __name__ == "__main__":
    unittest.main()