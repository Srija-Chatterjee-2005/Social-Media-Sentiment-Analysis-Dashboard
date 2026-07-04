import unittest

import pandas as pd

from src.xquik_export import normalize_xquik_export_columns


class XquikExportTest(unittest.TestCase):
    def test_maps_xquik_text_date_and_id_columns(self):
        normalized = normalize_xquik_export_columns(
            pd.DataFrame(
                [
                    {
                        "tweet_id": "t1",
                        "created_at": "2026-07-04T12:00:00Z",
                        "full_text": "Great update",
                    }
                ]
            )
        )

        self.assertEqual(normalized.loc[0, "text"], "Great update")
        self.assertEqual(normalized.loc[0, "date"], "2026-07-04T12:00:00Z")
        self.assertEqual(normalized.loc[0, "source_id"], "t1")
        self.assertEqual(normalized.loc[0, "platform"], "Xquik")

    def test_preserves_existing_platform_column(self):
        normalized = normalize_xquik_export_columns(
            pd.DataFrame([{"text": "Existing", "platform": "Twitter/X"}])
        )

        self.assertEqual(normalized.loc[0, "platform"], "Twitter/X")


if __name__ == "__main__":
    unittest.main()
