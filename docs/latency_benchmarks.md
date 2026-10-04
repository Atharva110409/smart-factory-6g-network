# 3GPP and ITU Latency & Reliability Benchmarks for Industrial Automation

This document provides reference communication benchmarks according to 3GPP and ITU standards for smart manufacturing, cyber-physical control systems, and industrial IoT.

## Benchmark Reference Table

| Use Case | End-to-End Latency Target | Service Reliability Target | Primary Industry Reference |
| :--- | :--- | :--- | :--- |
| **Tactile Interaction / Teleoperation** | 0.5 ms | 99.999% (5-nines) | 3GPP TS 22.104 / ITU-R M.2083 |
| **General URLLC Target** | ~1 ms | 99.999% (5-nines) | 3GPP Rel-15/16/17 URLLC core specification |
| **Electricity Distribution (High Voltage)** | 5 ms | 99.9999% (6-nines) | 3GPP TS 22.104 Smart Grid profile |
| **Intelligent Transport (Infra Backhaul)** | 10 ms | 99.9999% (6-nines) | 3GPP V2X / C-V2X architecture specs |
| **Discrete Automation (Motion Control / Robotics)** | 10 ms | 99.99% (4-nines) | 3GPP TS 22.104 Factory Automation |
| **Electricity Distribution (Medium Voltage)** | 25 ms | 99.9% (3-nines) | 3GPP TS 22.104 Distribution Automation |
| **Process Automation (Remote Closed-Loop Control)**| 50 ms | 99.9% (3-nines) | 3GPP TS 22.104 Process Automation |
| **Process Automation (Monitoring & Telemetry)** | 50 ms | 99.9% (3-nines) | 3GPP TS 22.104 Plant Monitoring |

### Sources
- **3GPP TS 22.104**: *Service requirements for cyber-physical control applications in vertical domains* (Rel-16/17/18).
- **Summarized in**: arXiv:2508.20205.
- **Supporting literature**: arXiv:2106.11825, arXiv:2509.10617, arXiv:2510.08080.

---

## Methodological Comparison Cautions & Boundary Conditions

### 1. Latency Measurement Semantics
- **Benchmark Definition**: 3GPP end-to-end latency is strictly defined as the time from when an application generates a message at the ingress service data unit (SDU) interface to when the message is correctly delivered at the egress SDU interface (one-way radio + transport latency).
- **Factory Dataset Metric (`Network_Latency_ms`)**: In real-world telemetry or industrial testbeds, recorded latency is frequently measured via ICMP/round-trip ping (RTT), application-level poll-response intervals, or network layer transport latency.
- **Rule**: If the exact physical layer probe mechanism cannot be confirmed as pure one-way user-plane latency, comparisons must be treated as **qualitative references** rather than direct one-to-one compliance thresholds.

### 2. Service Reliability vs. Packet Loss Percentage
- **Critical Principle**: **Do NOT equate packet loss % with service reliability %.**
- In 3GPP standards, reliability of $99.999\%$ means that $99.999\%$ of transmitted packets are delivered within the maximum permissible latency bound $T_{\text{max}}$.
- Unadjusted packet loss recorded in continuous telemetry logs reflects transport/socket drop rates or missing polling samples, but does not capture HARQ retransmissions, application-layer redundancy, or protocol timeout recovery mechanisms.
- All comparisons to 3GPP/ITU reliability ratings in reports and dashboard insights must be contextualized qualitatively unless exact packet-level transmission logs are available.
