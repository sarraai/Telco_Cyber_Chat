# Telco Domain Guidance

## Core entities (use correct terms)
- 4G EPC: MME, SGW, PGW, HSS
- 5G Core: AMF, SMF, UPF, AUSF, UDM, NRF, PCF
- IMS/VoIP: SIP, RTP, SBC, CSCF, ENUM/DNS

## Common security themes (high-level)
- Signaling abuse (SS7/Diameter), interconnect trust issues
- SIP fraud, toll fraud, registration abuse
- API exposure in 5G core and orchestration layers
- Misconfigurations in SBC, IMS, Kubernetes/Helm, CI/CD

## Output expectations
- Provide defensive guidance: detection, hardening, patching, monitoring.
- Avoid step-by-step offensive procedures.
- If user gives environment details (vendor, version, topology), tailor mitigations and checks.
