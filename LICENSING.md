# Licensing scope

## Code — Apache-2.0

UAV-Flow-authored code and modifications are licensed under [Apache-2.0](LICENSE),
except where an existing file-specific license applies. This includes executable
code, code examples, and software configuration; configuration files do not
become noncommercial merely because they use JSON, YAML, or TOML.

Project-authored files covered by these declarations do not need a repeated
per-file license header. Preserve existing third-party copyright, license, and
modification notices. When modifying Apache-licensed upstream files, retain or
add the required notice that the file was changed. Each independently distributed
component must carry its applicable license texts and scope declarations.

## Datasets, Images, and Documentation — CC BY-NC 4.0

UAV-Flow-authored datasets, images, and documentation prose are licensed under
[CC BY-NC 4.0](LICENSE-CONTENT), to the extent the project holds the relevant
rights. This covers our datasets as well as images and documentation, including:

- [UAV-Flow](https://huggingface.co/datasets/wangxiangyu0814/UAV-Flow) [UAV-Flow-Beta](https://huggingface.co/datasets/wangxiangyu0814/UAV-Flow-Beta)real-world
  flight data and [UAV-Flow-Sim](https://huggingface.co/datasets/wangxiangyu0814/UAV-Flow-Sim)
  simulated flight data, including project-authored RGB frames, language
  instructions, and trajectory logs.
- The root README and component guides maintained for UAV-Flow.
- `assets/`.
- `flowVLA/starVLA/config/dataset_statistics/`.
- `UAV-Flow-Eval/src/training_tasks/` task definitions.

Keep the supplied creator and copyright notices, link to the license, and
indicate modifications when sharing content. Use under this license must be
noncommercial. Code examples within documentation follow the applicable code
license instead.

## Existing licenses and external resources

- OpenVLA and StarVLA material retains its existing MIT license and attribution;
  see [OpenVLA-UAV/LICENSE](OpenVLA-UAV/LICENSE) and
  [flowVLA/LICENSE](flowVLA/LICENSE). Inherited documentation and images retain
  their original terms too.
  The bundled StarVLA license also includes repository-history clauses; its
  complete wording is retained rather than replaced with a generic MIT template.
- UnrealZoo and other bundled third-party notices remain in
  [UAV-Flow-Eval/LICENSE](UAV-Flow-Eval/LICENSE). Upstream evaluation reference
  JSON retains its Apache-2.0 grant. Project licenses do not remove these terms.
- Previously published material retains permissions already granted under earlier
  licenses. Applying CC BY-NC 4.0 does not revoke an existing Apache or MIT grant.
- Third-party datasets, simulator packages, scene assets, model weights, and
  dependencies follow their own terms. This policy includes our UAV-Flow datasets
  listed above; it does not relicense third-party material. Remote dataset cards
  and license metadata must be updated separately to reflect this declaration.

Preserve these scope declarations and the applicable license texts when
redistributing material. The evaluation package includes a local copy of the
content license for standalone distribution.
