import json
import sys
from datetime import datetime

import pytest
from gravity_tech.cli import run_complete_pipeline as cli
from gravity_tech.core.domain.entities import Candle


def test_cli_fetches_candles_via_data_connector(monkeypatch, tmp_path):
    called: dict[str, object] = {}

    def fake_fetch_candles(
        self,
        symbol: str,
        interval: str = "1h",
        limit: int = 500,
        start_date=None,
        end_date=None,
        **kwargs,
    ) -> list[Candle]:
        called["symbol"] = symbol
        called["interval"] = interval
        called["limit"] = limit
        base_dt = datetime(2024, 1, 1)
        return [
            Candle(
                timestamp=base_dt,
                open=1.0,
                high=2.0,
                low=0.5,
                close=1.5,
                volume=100.0,
            )
        ]

    class DummyPipeline:
        def __init__(self, candles: list[Candle]):
            self._candles = candles

        def analyze(self):
            class Result:
                def __init__(self, count: int):
                    self.count = count

                def to_dict(self) -> dict[str, object]:
                    return {"ok": True, "count": self.count}

            return Result(len(self._candles))

    def fake_build_pipeline_from_weights(*, candles: list[Candle], **kwargs):
        called["candles_count"] = len(candles)
        return DummyPipeline(candles)

    monkeypatch.setattr(cli.DataConnector, "fetch_candles", fake_fetch_candles)
    monkeypatch.setattr(cli, "build_pipeline_from_weights", fake_build_pipeline_from_weights)

    weight_paths = {
        "trend": tmp_path / "trend.json",
        "momentum": tmp_path / "momentum.json",
        "volatility": tmp_path / "volatility.json",
    }
    for path in weight_paths.values():
        path.write_text("{}", encoding="utf-8")

    output_path = tmp_path / "pipeline.json"
    argv = [
        "prog",
        "--symbol",
        "TEST",
        "--interval",
        "4h",
        "--limit",
        "25",
        "--trend-weights",
        str(weight_paths["trend"]),
        "--momentum-weights",
        str(weight_paths["momentum"]),
        "--volatility-weights",
        str(weight_paths["volatility"]),
        "--output",
        str(output_path),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    cli.main()

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert payload["count"] == 1
    assert called["symbol"] == "TEST"
    assert called["interval"] == "4h"
    assert called["limit"] == 25
    assert called["candles_count"] == 1


def test_cli_exits_on_connector_failure(monkeypatch, tmp_path):
    def fake_fetch_candles(*args, **kwargs):
        raise RuntimeError("connector down")

    monkeypatch.setattr(cli.DataConnector, "fetch_candles", fake_fetch_candles)

    weight_paths = {
        "trend": tmp_path / "trend.json",
        "momentum": tmp_path / "momentum.json",
        "volatility": tmp_path / "volatility.json",
    }
    for path in weight_paths.values():
        path.write_text("{}", encoding="utf-8")

    argv = [
        "prog",
        "--symbol",
        "FAIL",
        "--trend-weights",
        str(weight_paths["trend"]),
        "--momentum-weights",
        str(weight_paths["momentum"]),
        "--volatility-weights",
        str(weight_paths["volatility"]),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit):
        cli.main()
