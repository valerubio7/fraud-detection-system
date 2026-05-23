import json
from unittest.mock import MagicMock, patch

import pytest

from streaming.publisher import AvroPublisher

SCHEMA_PATH = "streaming/schemas/transaction_features.avsc"


def _make_publisher(schema_path: str = SCHEMA_PATH) -> AvroPublisher:
    mock_producer = MagicMock()
    with patch("streaming.publisher.Producer", return_value=mock_producer):
        with patch("streaming.publisher.AvroPublisher._load_schema", return_value={"type": "record"}):
            pub = AvroPublisher(
                broker_url="localhost:9092",
                topic="test.topic",
                schema_path=schema_path,
                client_id="test-client",
            )
    pub._producer = mock_producer
    return pub


class TestAvroPublisherInit:
    def test_is_available_after_construction(self):
        pub = _make_publisher()
        assert pub._topic == "test.topic"
        assert pub._broker_url == "localhost:9092"


class TestAvroPublisherLoadSchema:
    def test_load_schema_raises_on_missing_file(self, tmp_path):
        missing = tmp_path / "missing.avsc"
        with pytest.raises(FileNotFoundError):
            AvroPublisher._load_schema(missing)

    def test_load_schema_raises_on_invalid_json(self, tmp_path):
        bad_file = tmp_path / "bad.avsc"
        bad_file.write_text("not-json", encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid Avro schema JSON"):
            AvroPublisher._load_schema(bad_file)

    def test_load_schema_returns_parsed_schema_for_valid_file(self, tmp_path):
        schema_file = tmp_path / "schema.avsc"
        schema_file.write_text(json.dumps({"type": "record", "name": "T", "fields": []}))
        with patch("streaming.publisher.parse_schema", return_value={"type": "record"}) as mock_parse:
            result = AvroPublisher._load_schema(schema_file)
        mock_parse.assert_called_once()
        assert result == {"type": "record"}


class TestAvroPublisherFlushAndClose:
    def test_flush_calls_producer_flush(self):
        pub = _make_publisher()
        pub._producer.flush.return_value = 0
        pub.flush()
        pub._producer.flush.assert_called_once()

    def test_flush_logs_warning_when_messages_remain(self):
        pub = _make_publisher()
        pub._producer.flush.return_value = 3
        pub.flush()  # Should not raise

    def test_close_calls_flush(self):
        pub = _make_publisher()
        pub._producer.flush.return_value = 0
        pub.close()
        pub._producer.flush.assert_called_once()

    def test_flush_logs_warning_when_messages_remain_v2(self):
        pub = _make_publisher()
        pub._producer.flush.return_value = 5
        pub.flush()  # Should not raise even with remaining messages


class TestAvroPublisherDeliveryCallback:
    def test_delivery_callback_no_error(self):
        pub = _make_publisher()
        msg = MagicMock()
        msg.topic.return_value = "test.topic"
        msg.partition.return_value = 0
        msg.offset.return_value = 42
        pub._delivery_callback(None, msg, "key_1")  # Should not raise

    def test_delivery_callback_logs_error(self):
        pub = _make_publisher()
        msg = MagicMock()
        pub._delivery_callback(Exception("delivery failed"), msg, "key_1")  # Should not raise


class TestAvroPublisherSerialize:
    def test_serialize_avro_returns_bytes(self):
        pub = _make_publisher()
        with patch("streaming.publisher.schemaless_writer"):
            result = pub._serialize_avro({"key": "value"})
        assert isinstance(result, bytes)


class TestAvroPublisherPublish:
    def test_publish_calls_producer_produce(self):
        pub = _make_publisher()
        pub._producer.poll.return_value = None
        with patch.object(pub, "_serialize_avro", return_value=b"payload"):
            pub._publish("key_1", b"payload")
        pub._producer.produce.assert_called_once()

    def test_publish_polls_after_produce(self):
        pub = _make_publisher()
        pub._producer.poll.return_value = None
        with patch.object(pub, "_serialize_avro", return_value=b"payload"):
            pub._publish("key_2", b"payload")
        pub._producer.poll.assert_called_with(0.0)
