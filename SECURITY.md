# Security Policy

## Supported versions

The latest tagged release and the `main` branch receive security fixes during the research-release phase.

## Reporting a vulnerability

Do not open a public issue for vulnerabilities involving arbitrary code execution, unsafe checkpoint loading, dependency compromise, credential exposure, or denial of service. Email `artur@msctechnology.com.br` with:

- affected version or commit;
- minimal reproduction;
- impact assessment;
- suggested mitigation, when available.

The public checkpoint path uses `safetensors`. The repository does not endorse loading untrusted pickle checkpoints.
