from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
import tempfile
import threading
from typing import Iterator
import unittest

import numpy as np

from Utils.session_edits import SessionEditStore, check_session_edits


class SessionEditsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = (
            Path(self.temporary_directory.name)
            / "m15"
            / "session_slices.sqlite3"
        )
        self.session_edit = check_session_edits(
            self.database_path,
            "260716",
            3,
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    @contextmanager
    def _database_connection(self) -> Iterator[sqlite3.Connection]:
        database_connection = sqlite3.connect(self.database_path)
        try:
            with database_connection:
                yield database_connection
        finally:
            database_connection.close()

    def test_schema_has_exactly_one_five_column_session_table(self) -> None:
        with self._database_connection() as database_connection:
            table_names = {
                row[0]
                for row in database_connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
            column_rows = database_connection.execute(
                "PRAGMA table_info(session_edits)"
            ).fetchall()

        self.assertIn("session_edits", table_names)
        self.assertNotIn("session_delete_frames", table_names)
        self.assertEqual(
            [row[1] for row in column_rows],
            [
                "date",
                "session_id",
                "delete_frames",
                "interp_start_end",
                "interp_frames_between",
            ],
        )
        self.assertEqual(
            [(row[1], row[5]) for row in column_rows if row[5] > 0],
            [("date", 1), ("session_id", 2)],
        )

    def test_session_edits_can_be_created_next_to_old_session_slices(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "m15" / "edits.sqlite3"
            database_path.parent.mkdir(parents=True)
            database_connection = sqlite3.connect(database_path)
            try:
                with database_connection:
                    database_connection.execute(
                        "CREATE TABLE session_slices (date TEXT, session_id TEXT)"
                    )
            finally:
                database_connection.close()

            check_session_edits(database_path, "260716", 3)

            database_connection = sqlite3.connect(database_path)
            try:
                table_names = {
                    row[0]
                    for row in database_connection.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table'"
                    )
                }
            finally:
                database_connection.close()

        self.assertIn("session_slices", table_names)
        self.assertIn("session_edits", table_names)
        self.assertNotIn("session_delete_frames", table_names)

    def test_frame_and_range_deletes_append_to_one_flat_list(self) -> None:
        self.assertEqual(self.session_edit.deleteByFrame([2, 4]), (True, None))
        self.assertEqual(self.session_edit.deleteByFrame([4, 6]), (True, None))
        self.assertEqual(self.session_edit.deleteByRange(2, -2), (True, None))

        self.assertEqual(
            self.session_edit.getDelete(),
            [2, 4, 6, 0, 1, -2, -1],
        )

        with self._database_connection() as database_connection:
            rows = database_connection.execute(
                """
                SELECT date,
                       session_id,
                       delete_frames,
                       interp_start_end,
                       interp_frames_between
                FROM session_edits
                """
            ).fetchall()

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][:2], ("260716", "3"))
        self.assertEqual(json.loads(rows[0][2]), [2, 4, 6, 0, 1, -2, -1])
        self.assertEqual(rows[0][3:], (None, None))

    def test_repeating_a_saved_delete_is_idempotent(self) -> None:
        self.assertEqual(self.session_edit.deleteByFrame([5, 6]), (True, None))
        self.assertEqual(self.session_edit.deleteByFrame([5, 6]), (True, None))
        self.assertEqual(self.session_edit.getDelete(), [5, 6])

    def test_concurrent_appends_do_not_lose_frames(self) -> None:
        start_barrier = threading.Barrier(2)

        def append_frame(frame: int) -> tuple[bool, str | None]:
            edit_store = SessionEditStore(self.database_path)
            start_barrier.wait()
            return edit_store.add_delete_frames("260716", "3", [frame])

        with ThreadPoolExecutor(max_workers=2) as thread_pool:
            results = list(thread_pool.map(append_frame, [10, 20]))

        self.assertEqual(results, [(True, None), (True, None)])
        self.assertEqual(set(self.session_edit.getDelete()), {10, 20})
        self.assertEqual(len(self.session_edit.getDelete()), 2)

    def test_get_session_info_applies_one_flat_delete(self) -> None:
        self.session_edit.deleteByFrame([2, 4, 6])
        self.session_edit.deleteByRange(2, -2)

        (changed, operations), edited_target = self.session_edit.getSessionInfo(
            target=np.arange(10)
        )

        self.assertTrue(changed)
        self.assertEqual(operations, ["delete"])
        np.testing.assert_array_equal(edited_target, np.array([3, 5, 7]))

    def test_one_interpolation_is_upserted_in_the_same_row(self) -> None:
        self.session_edit.deleteByFrame([7])
        self.assertEqual(self.session_edit.interp(0, 3, 2), (True, None))
        self.assertEqual(self.session_edit.getInterp(), [(0, 3, 2)])

        self.assertEqual(self.session_edit.interp(1, 4, 1), (True, None))
        self.assertEqual(self.session_edit.getInterp(), [(1, 4, 1)])
        self.assertEqual(self.session_edit.getDelete(), [7])

        with self._database_connection() as database_connection:
            row_count = database_connection.execute(
                "SELECT COUNT(*) FROM session_edits"
            ).fetchone()[0]
            stored_pair = database_connection.execute(
                "SELECT interp_start_end FROM session_edits"
            ).fetchone()[0]

        self.assertEqual(row_count, 1)
        self.assertEqual(json.loads(stored_pair), [1, 4])

    def test_frames_between_excludes_the_two_anchor_frames(self) -> None:
        self.session_edit.interp(0, 3, 2)

        (changed, operations), edited_target = self.session_edit.getSessionInfo(
            target=np.array([0.0, 1.0, 4.0, 9.0])
        )

        self.assertTrue(changed)
        self.assertEqual(operations, ["interpolate"])
        np.testing.assert_allclose(edited_target, np.array([0.0, 3.0, 6.0, 9.0]))

        self.session_edit.interp(0, 3, 0)
        (_, _), edited_target = self.session_edit.getSessionInfo(
            target=np.array([0.0, 1.0, 4.0, 9.0])
        )
        np.testing.assert_allclose(edited_target, np.array([0.0, 9.0]))

    def test_delete_after_interpolation_is_rejected_without_mutation(self) -> None:
        self.session_edit.deleteByFrame([2])
        self.session_edit.interp(0, 3, 1)

        success, message = self.session_edit.deleteByRange(1, -1)

        self.assertFalse(success)
        self.assertIn("clear the saved interpolation", message)
        self.assertEqual(self.session_edit.getDelete(), [2])
        self.assertEqual(self.session_edit.getInterp(), [(0, 3, 1)])

    def test_clear_interpolation_keeps_delete_or_removes_empty_row(self) -> None:
        self.session_edit.deleteByFrame([2])
        self.session_edit.interp(0, 3, 1)
        self.assertEqual(self.session_edit.clearInterps(), (True, None))
        self.assertEqual(self.session_edit.getDelete(), [2])
        self.assertEqual(self.session_edit.getInterp(), [])
        self.assertEqual(self.session_edit.checkSaved(), (True, None))

        interpolation_only = check_session_edits(
            self.database_path,
            "260717",
            1,
        )
        interpolation_only.interp(0, 2, 0)
        self.assertEqual(interpolation_only.clearInterps(), (True, None))
        self.assertFalse(interpolation_only.checkSaved()[0])

    def test_invalid_delete_input_does_not_create_a_row(self) -> None:
        self.assertFalse(self.session_edit.deleteByFrame([])[0])
        self.assertFalse(self.session_edit.deleteByFrame([2, 2])[0])
        self.assertFalse(self.session_edit.deleteByFrame([2.5])[0])
        self.assertFalse(self.session_edit.deleteByFrame([True])[0])
        self.assertFalse(self.session_edit.deleteByFrame([np.bool_(True)])[0])
        self.assertFalse(self.session_edit.deleteByRange(1.5, -1)[0])
        self.assertFalse(self.session_edit.interp(0.5, 3, 1)[0])
        self.assertFalse(self.session_edit.checkSaved()[0])

    def test_replace_rejects_sequential_delete_groups_without_mutation(self) -> None:
        self.session_edit.deleteByFrame([9])
        edit_store = SessionEditStore(self.database_path)

        success, message = edit_store.replace_edits(
            "260716",
            "3",
            [
                {"operation": "delete", "frames": [1]},
                {"operation": "delete", "frames": [1]},
            ],
        )

        self.assertFalse(success)
        self.assertIn("multiple sequential delete groups", message)
        self.assertEqual(self.session_edit.getDelete(), [9])

    def test_sessions_are_isolated(self) -> None:
        other_session = check_session_edits(
            self.database_path,
            "260716",
            4,
        )
        self.session_edit.deleteByFrame([1])
        other_session.deleteByFrame([8])

        self.assertEqual(self.session_edit.getDelete(), [1])
        self.assertEqual(other_session.getDelete(), [8])


class LegacySessionEditsSchemaTests(unittest.TestCase):
    def test_legacy_ordered_schema_is_rejected_without_modification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "m15" / "edits.sqlite3"
            database_path.parent.mkdir(parents=True)
            database_connection = sqlite3.connect(database_path)
            try:
                with database_connection:
                    database_connection.execute(
                        """
                        CREATE TABLE session_edits (
                            edit_id INTEGER PRIMARY KEY,
                            mouse_id TEXT,
                            date TEXT,
                            session_id TEXT,
                            edit_order INTEGER,
                            operation TEXT,
                            interp_start_frame INTEGER,
                            interp_end_frame INTEGER,
                            interp_frames_between INTEGER
                        )
                        """
                    )
                    database_connection.execute(
                        """
                        CREATE TABLE session_delete_frames (
                            edit_id INTEGER,
                            frame_order INTEGER,
                            frame_index INTEGER
                        )
                        """
                    )
            finally:
                database_connection.close()

            with self.assertRaisesRegex(RuntimeError, "legacy ordered"):
                check_session_edits(database_path, "260716", 3)

            database_connection = sqlite3.connect(database_path)
            try:
                table_names = {
                    row[0]
                    for row in database_connection.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table'"
                    )
                }
            finally:
                database_connection.close()
            self.assertIn("session_edits", table_names)
            self.assertIn("session_delete_frames", table_names)

    def test_five_columns_without_primary_key_are_not_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "m15" / "edits.sqlite3"
            database_path.parent.mkdir(parents=True)
            database_connection = sqlite3.connect(database_path)
            try:
                with database_connection:
                    database_connection.execute(
                        """
                        CREATE TABLE session_edits (
                            date TEXT NOT NULL,
                            session_id TEXT NOT NULL,
                            delete_frames TEXT NOT NULL,
                            interp_start_end TEXT,
                            interp_frames_between INTEGER
                        )
                        """
                    )
            finally:
                database_connection.close()

            with self.assertRaisesRegex(RuntimeError, "Could not initialize"):
                check_session_edits(database_path, "260716", 3)

            database_connection = sqlite3.connect(database_path)
            try:
                primary_key_columns = [
                    row[1]
                    for row in database_connection.execute(
                        "PRAGMA table_info(session_edits)"
                    )
                    if row[5] > 0
                ]
            finally:
                database_connection.close()

            self.assertEqual(primary_key_columns, [])


if __name__ == "__main__":
    unittest.main()
