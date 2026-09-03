from slime.ray.utils import add_default_ray_env_vars
from slime.utils.external_utils.command_utils import _build_no_proxy
from slime.utils.misc import get_current_node_ip


def test_current_node_ip_respects_explicit_host_without_consulting_ray(monkeypatch):
    monkeypatch.setenv("SLIME_HOST_IP", "[::1]")

    assert get_current_node_ip() == "::1"


def test_default_ray_env_forwards_internal_network_overrides(monkeypatch):
    monkeypatch.setenv("SLIME_HOST_IP", "127.0.0.1")
    monkeypatch.setenv("no_proxy", "localhost,127.0.0.1")
    monkeypatch.setenv("NO_PROXY", "localhost,127.0.0.1")

    env = add_default_ray_env_vars()

    assert env["SLIME_HOST_IP"] == "127.0.0.1"
    assert env["no_proxy"] == "localhost,127.0.0.1"
    assert env["NO_PROXY"] == "localhost,127.0.0.1"


def test_ray_job_no_proxy_preserves_scheduler_and_render_hosts(monkeypatch):
    monkeypatch.setenv("no_proxy", "0.0.0.0,localhost,render.internal")
    monkeypatch.setenv("NO_PROXY", ".corp.example.com,render.internal")
    monkeypatch.setenv("SLIME_HOST_IP", "127.0.0.1")

    entries = _build_no_proxy("10.0.0.8").split(",")

    assert entries[:3] == ["127.0.0.1", "localhost", "::1"]
    assert "0.0.0.0" in entries
    assert "render.internal" in entries
    assert ".corp.example.com" in entries
    assert "10.0.0.8" in entries
    assert len(entries) == len(set(entries))
