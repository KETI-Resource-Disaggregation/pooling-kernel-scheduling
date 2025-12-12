#!/usr/bin/env python3
"""
Profiler Client - pooling-workload-profiler API 클라이언트

kernel-scheduling에서 workload-profiler의 프로파일 정보를 조회하기 위한 모듈

Usage:
    from profiler_client import ProfilerClient

    client = ProfilerClient("http://localhost:7070")
    profile = client.get_profile("resnet50")
    colocation = client.get_colocation_score("resnet50", "bert")
"""

import os
import json
import urllib.request
import urllib.error
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# ==================== Configuration ====================
PROFILER_URL = os.environ.get("PROFILER_URL", "http://localhost:7070")

class KernelType(Enum):
    UNKNOWN = 0
    COMPUTE_BOUND = 1
    MEMORY_BOUND = 2
    MIXED = 3

@dataclass
class WorkloadProfile:
    workload_id: str
    kernel_type: KernelType
    arithmetic_intensity: float
    compute_ratio: float
    memory_ratio: float
    mixed_ratio: float
    description: str
    source: str

    @classmethod
    def from_dict(cls, workload_id: str, data: Dict) -> 'WorkloadProfile':
        return cls(
            workload_id=workload_id,
            kernel_type=KernelType[data.get("kernel_type", "UNKNOWN")],
            arithmetic_intensity=float(data.get("arithmetic_intensity", 0)),
            compute_ratio=float(data.get("compute_ratio", 0.33)),
            memory_ratio=float(data.get("memory_ratio", 0.33)),
            mixed_ratio=float(data.get("mixed_ratio", 0.34)),
            description=data.get("description", ""),
            source=data.get("source", "unknown")
        )

@dataclass
class ColocationRecommendation:
    workload_a: str
    workload_b: str
    type_a: KernelType
    type_b: KernelType
    score: float
    recommendation: str  # "good", "moderate", "poor"
    sm_partition: Dict[str, int]

class ProfilerClient:
    """REST API client for pooling-workload-profiler"""

    def __init__(self, base_url: str = None):
        self.base_url = (base_url or PROFILER_URL).rstrip("/")
        self._cache: Dict[str, WorkloadProfile] = {}

    def _request(self, method: str, path: str, data: Dict = None) -> Dict:
        """Make HTTP request to profiler API"""
        url = f"{self.base_url}{path}"
        headers = {"Content-Type": "application/json"}

        try:
            if method == "GET":
                req = urllib.request.Request(url, headers=headers)
            else:
                body = json.dumps(data or {}).encode("utf-8")
                req = urllib.request.Request(url, data=body, headers=headers, method=method)

            with urllib.request.urlopen(req, timeout=5) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as e:
            return {"ok": False, "error": f"Connection failed: {e}"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def health_check(self) -> bool:
        """Check if profiler service is available"""
        result = self._request("GET", "/health")
        return result.get("ok", False)

    def get_profile(self, workload_id: str, use_cache: bool = True) -> Optional[WorkloadProfile]:
        """Get profile for a workload"""
        # Check cache first
        if use_cache and workload_id in self._cache:
            return self._cache[workload_id]

        result = self._request("GET", f"/profile/{workload_id}")
        if result.get("ok") and "profile" in result:
            profile = WorkloadProfile.from_dict(workload_id, result["profile"])
            self._cache[workload_id] = profile
            return profile

        return None

    def request_profile(self, workload_id: str, model: str = None) -> Optional[WorkloadProfile]:
        """Request profiling for a workload"""
        data = {"workload_id": workload_id}
        if model:
            data["model"] = model

        result = self._request("POST", "/profile", data)
        if result.get("ok") and "profile" in result:
            profile = WorkloadProfile.from_dict(workload_id, result["profile"])
            self._cache[workload_id] = profile
            return profile

        return None

    def register_profile(self, workload_id: str, kernel_type: str,
                        arithmetic_intensity: float = 0,
                        description: str = "") -> bool:
        """Manually register a workload profile"""
        data = {
            "workload_id": workload_id,
            "kernel_type": kernel_type,
            "arithmetic_intensity": arithmetic_intensity,
            "description": description
        }
        result = self._request("POST", "/register", data)
        return result.get("ok", False)

    def get_colocation_score(self, workload_a: str, workload_b: str) -> Optional[ColocationRecommendation]:
        """Get co-location recommendation for two workloads"""
        result = self._request("GET", f"/colocation?a={workload_a}&b={workload_b}")

        if result.get("ok"):
            return ColocationRecommendation(
                workload_a=workload_a,
                workload_b=workload_b,
                type_a=KernelType[result.get("workload_a", {}).get("type", "UNKNOWN")],
                type_b=KernelType[result.get("workload_b", {}).get("type", "UNKNOWN")],
                score=float(result.get("colocation_score", 0)),
                recommendation=result.get("recommendation", "unknown"),
                sm_partition=result.get("sm_partition", {})
            )

        return None

    def get_all_profiles(self) -> Dict[str, WorkloadProfile]:
        """Get all registered profiles"""
        result = self._request("GET", "/profiles")
        if result.get("ok") and "profiles" in result:
            profiles = {}
            for wid, data in result["profiles"].items():
                profiles[wid] = WorkloadProfile.from_dict(wid, data)
                self._cache[wid] = profiles[wid]
            return profiles
        return {}

    def clear_cache(self):
        """Clear local profile cache"""
        self._cache.clear()


# ==================== Convenience Functions ====================
_default_client: Optional[ProfilerClient] = None

def get_client() -> ProfilerClient:
    """Get or create default profiler client"""
    global _default_client
    if _default_client is None:
        _default_client = ProfilerClient()
    return _default_client

def get_kernel_type(workload_id: str) -> KernelType:
    """Quick lookup for kernel type"""
    client = get_client()
    profile = client.get_profile(workload_id)
    if profile:
        return profile.kernel_type
    return KernelType.UNKNOWN

def should_colocate(workload_a: str, workload_b: str) -> Tuple[bool, float]:
    """
    Quick check if two workloads should be co-located

    Returns:
        (should_colocate: bool, score: float)
    """
    client = get_client()
    rec = client.get_colocation_score(workload_a, workload_b)
    if rec:
        return (rec.score > 0.5, rec.score)
    return (False, 0.0)

def get_sm_recommendation(workload_a: str, workload_b: str, total_sms: int = 84) -> Dict[str, int]:
    """Get recommended SM partition for two workloads"""
    client = get_client()
    rec = client.get_colocation_score(workload_a, workload_b)
    if rec and rec.sm_partition:
        return rec.sm_partition

    # Default: equal split
    half = total_sms // 2
    return {"workload_a": half, "workload_b": total_sms - half}


# ==================== CLI ====================
def main():
    """CLI for testing profiler client"""
    import sys

    client = ProfilerClient()

    if not client.health_check():
        print("Error: Cannot connect to profiler service")
        print(f"URL: {client.base_url}")
        sys.exit(1)

    print(f"Connected to profiler: {client.base_url}")
    print()

    # Get all profiles
    profiles = client.get_all_profiles()
    print(f"Available profiles ({len(profiles)}):")
    for wid, profile in profiles.items():
        print(f"  {wid}: {profile.kernel_type.name} (AI={profile.arithmetic_intensity:.1f})")
    print()

    # Example co-location check
    if len(sys.argv) >= 3:
        a, b = sys.argv[1], sys.argv[2]
    else:
        a, b = "resnet50", "bert"

    rec = client.get_colocation_score(a, b)
    if rec:
        print(f"Co-location: {a} + {b}")
        print(f"  Types: {rec.type_a.name} + {rec.type_b.name}")
        print(f"  Score: {rec.score:.2f} ({rec.recommendation})")
        print(f"  SM Partition: {rec.sm_partition}")

if __name__ == "__main__":
    main()
