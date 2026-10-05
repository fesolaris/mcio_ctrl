# mcio_ctrl

### [MCio mod](https://github.com/twoturtles/MCio) | [mcio_ctrl](https://github.com/twoturtles/mcio_ctrl) | [Documentation](https://github.com/twoturtles/mcio_ctrl/wiki) | [Discord](https://discord.gg/PBfdc27h4q)

Python library for [MCio](https://github.com/twoturtles/MCio), a Minecraft Fabric mod for AI agent development. Includes [Gymnasium](https://gymnasium.farama.org/) environments for reinforcement learning.

## Features

* Install and launch Minecraft with the MCio mod, create worlds, and run the game from Python
* Faster than real-time performance (>13x on an M3 laptop)
* Gymnasium environments with MineRL 1.0 compatible actions/observations
* Interactive GUI for human control via standard Minecraft controls (human-in-the-loop planned)
* Fully type-hinted
* [VPT and STEVE-1 support](https://github.com/jxiong21029/mcio-vpt-example) on modern Minecraft with [Sodium](https://modrinth.com/mod/sodium)

## Performance

* `extras/bench_sync.py` measures SYNC throughput end to end (single-player and a LAN duel pair driven
  by `MCioMultiEnv`) and prints one `BENCH {json}` line per run; `--profile` adds a cProfile breakdown.
* **Frame skip**: pass `options={"send_frame": False}` to `step()` / `begin_step()` (or the matching
  entry of `MCioMultiEnv.step`) on steps whose frame the caller discards, e.g. the non-final tick of an
  action repeat. Minecraft then skips the framebuffer readback and sends the observation without a
  frame; the environment keeps the previous frame. `ObservationPacket.has_frame()` reports whether an
  observation carries one. Unused, this changes nothing: the field is not encoded on the wire unless
  requested. A client that requests it needs the matching MCio mod build (an older mod rejects the
  unknown field), which is why it is opt-in.
* Profiling the mod side: start the instance with `MCIO_PROFILE=true` (see the MCio README).

## Links

* [Documentation / Wiki](https://github.com/twoturtles/mcio_ctrl/wiki)
* [MCio mod](https://github.com/twoturtles/MCio) ([Modrinth](https://modrinth.com/mod/mcio))
* [PyPI](https://pypi.org/project/mcio_ctrl/)
