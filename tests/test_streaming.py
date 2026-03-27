"""Tests for NDJSON response streaming."""

import json


class TestNdjsonGenerator:
    """Test NDJSON batch streaming."""

    def test_basic_streaming(self):
        from opencut.core.streaming import ndjson_generator

        items = [{"id": i, "text": f"item {i}"} for i in range(5)]
        lines = list(ndjson_generator(items, chunk_size=3))

        # Should have: header + 2 batches + done
        assert len(lines) == 4

        header = json.loads(lines[0])
        assert header["type"] == "header"
        assert header["total"] == 5

        batch1 = json.loads(lines[1])
        assert batch1["type"] == "batch"
        assert len(batch1["items"]) == 3
        assert batch1["sent"] == 3

        batch2 = json.loads(lines[2])
        assert batch2["type"] == "batch"
        assert len(batch2["items"]) == 2
        assert batch2["sent"] == 5

        done = json.loads(lines[3])
        assert done["type"] == "done"
        assert done["total_sent"] == 5

    def test_empty_items(self):
        from opencut.core.streaming import ndjson_generator

        lines = list(ndjson_generator([], chunk_size=10))
        assert len(lines) == 2  # header + done

        done = json.loads(lines[1])
        assert done["total_sent"] == 0

    def test_exact_chunk_size(self):
        from opencut.core.streaming import ndjson_generator

        items = [{"i": i} for i in range(10)]
        lines = list(ndjson_generator(items, chunk_size=5))
        # header + 2 batches + done
        assert len(lines) == 4


class TestNdjsonItemGenerator:
    """Test per-item NDJSON streaming."""

    def test_per_item(self):
        from opencut.core.streaming import ndjson_item_generator

        items = [{"id": 1}, {"id": 2}, {"id": 3}]
        lines = list(ndjson_item_generator(items))

        # header + 3 items + done = 5 lines
        assert len(lines) == 5

        header = json.loads(lines[0])
        assert header["type"] == "header"
        assert header["total"] == 3

        item1 = json.loads(lines[1])
        assert item1["id"] == 1

        done = json.loads(lines[4])
        assert done["type"] == "done"
        assert done["total_sent"] == 3

    def test_all_lines_are_valid_json(self):
        from opencut.core.streaming import ndjson_item_generator

        items = [{"x": i} for i in range(20)]
        for line in ndjson_item_generator(items):
            parsed = json.loads(line.strip())
            assert isinstance(parsed, dict)


class TestNdjsonProgressGenerator:
    """Test progress-aware streaming."""

    def test_progress_generator(self):
        from opencut.core.streaming import ndjson_progress_generator

        def gen():
            for i in range(3):
                yield {"id": i}, (i + 1) * 33

        lines = list(ndjson_progress_generator(gen, total_hint=3))
        assert len(lines) == 5  # header + 3 items + done

        item = json.loads(lines[1])
        assert item["id"] == 0
        assert item["_progress"] == 33
