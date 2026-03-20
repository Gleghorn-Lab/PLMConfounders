"""
Sequence and data clustering utilities (CD-HIT / MMseqs2).
"""
import os
import shutil
import subprocess
import psutil
import torch
from collections import defaultdict
from typing import Any, Dict, List, Tuple
from tqdm.auto import tqdm


class ClusteringConfig:
    def __init__(
        self,
        method: str = "mmseqs2",
        cluster_percentage: float = 0.5,
        coverage: float = 0.8,
        n: int = 3,
        memory_percentage: float = 0.5,
        cache_suffix: str = "",
        base_path: str = "data",
        identifier: str = "unknown",
    ):
        self.method = method
        self.cluster_percentage = cluster_percentage
        self.coverage = coverage
        self.n = n
        self.memory_percentage = memory_percentage
        self.cache_suffix = cache_suffix
        self.base_path = base_path
        self.identifier = identifier


class ProteinSequenceClusterer:
    def __init__(self, config: ClusteringConfig) -> None:
        self.config = config

    def _run_docker_command(self, cmd: List[str], **kwargs: Any) -> subprocess.CompletedProcess:
        try:
            result = subprocess.run(cmd, **kwargs)
            if result.returncode == 0:
                return result
        except Exception:
            pass

        if os.name == "nt":
            return subprocess.run(cmd, **kwargs)

        return subprocess.run(["sudo"] + cmd, **kwargs)

    def _validate_paths_under_cwd(self, *paths: str) -> None:
        cwd = os.path.abspath(os.getcwd())
        for p in paths:
            ap = os.path.abspath(p)
            if not (ap == cwd or ap.startswith(cwd + os.sep)):
                raise ValueError(
                    "Path must be under current working directory for docker volume mount. "
                    f"cwd={cwd!r}, path={ap!r}"
                )

    def _path_in_container(self, local_path: str) -> str:
        self._validate_paths_under_cwd(local_path)
        rel = os.path.relpath(
            os.path.abspath(local_path),
            start=os.path.abspath(os.getcwd()),
        )
        return rel.replace(os.sep, "/")

    def _write_fasta_if_missing(self, seq_dict: Dict[str, str]) -> str:
        os.makedirs(self.config.base_path, exist_ok=True)
        fasta_path = os.path.join(
            self.config.base_path,
            f"{self.config.identifier}{self.config.cache_suffix}.fasta",
        )
        if not os.path.exists(fasta_path):
            with open(fasta_path, "w") as f:
                for seq_id, seq in seq_dict.items():
                    f.write(f">{seq_id}\n{seq}\n")
        return fasta_path

    def _compute_resources(self) -> Tuple[int, int]:
        num_cpu = os.cpu_count() - 4 if os.cpu_count() and os.cpu_count() > 4 else 1
        memory_max = int(
            self.config.memory_percentage * psutil.virtual_memory().total / 1024 / 1024
        )
        return num_cpu, memory_max

    def cdhit(self, seq_dict: Dict[str, str]) -> Dict[str, List[str]]:
        fasta_path = self._write_fasta_if_missing(seq_dict)
        num_cpu, memory_max = self._compute_resources()
        output_path = os.path.join(
            self.config.base_path,
            f"output_{self.config.identifier}_{self.config.cluster_percentage}{self.config.cache_suffix}",
        )
        cluster_file = f"{output_path}.clstr"
        self._validate_paths_under_cwd(fasta_path, output_path, cluster_file)

        if not os.path.exists(cluster_file):
            docker_image = "cd-hit"
            self._run_docker_command(
                [
                    "docker",
                    "run",
                    "--rm",
                    "-v",
                    f"{os.getcwd()}:/data",
                    "-w",
                    "/data",
                    docker_image,
                    "cd-hit",
                    "-i",
                    self._path_in_container(fasta_path),
                    "-o",
                    self._path_in_container(output_path),
                    "-d",
                    "0",
                    "-c",
                    str(self.config.cluster_percentage),
                    "-n",
                    str(self.config.n),
                    "-T",
                    str(num_cpu),
                    "-M",
                    str(memory_max),
                    "-aL",
                    str(self.config.coverage),
                ],
                check=True,
            )

        cluster_dict = defaultdict(list)
        with open(cluster_file, "r") as f:
            current_cluster = None
            for line in tqdm(f, desc="Reading CD-HIT cluster file"):
                if line.startswith(">"):
                    current_cluster = line.split("Cluster")[1].strip()
                else:
                    seq_id = line.split(">")[1].split("...")[0].strip()
                    cluster_dict[current_cluster].append(seq_id)
        return cluster_dict

    def mmseqs2(self, seq_dict: Dict[str, str]) -> Dict[str, List[str]]:
        fasta_path = self._write_fasta_if_missing(seq_dict)
        num_cpu, memory_max = self._compute_resources()
        output_prefix = os.path.join(
            self.config.base_path,
            f"mmseqs_{self.config.identifier}_{self.config.cluster_percentage}{self.config.cache_suffix}",
        )
        cluster_tsv = f"{output_prefix}_cluster.tsv"
        tmp_dir = os.path.join(
            self.config.base_path,
            f"tmp_{self.config.identifier}{self.config.cache_suffix}",
        )
        os.makedirs(tmp_dir, exist_ok=True)
        self._validate_paths_under_cwd(fasta_path, cluster_tsv, tmp_dir, self.config.base_path)

        if not os.path.exists(cluster_tsv):
            use_gpu = torch.cuda.is_available()

            mmseqs_args = [
                "easy-cluster",
                fasta_path,
                output_prefix,
                tmp_dir,
                "--min-seq-id",
                str(self.config.cluster_percentage),
                "-c",
                str(self.config.coverage),
                "--threads",
                str(num_cpu),
                "--cov-mode",
                "0",
                "-s",
                "7.5",
                "--split-memory-limit",
                f"{memory_max}M",
            ]
            if use_gpu:
                mmseqs_args.extend(["--gpu", "1"])

            # Try native mmseqs first (e.g. when running inside a container)
            if shutil.which("mmseqs"):
                print("Running MMseqs2 natively...")
                subprocess.run(["mmseqs"] + mmseqs_args, check=True)
            else:
                # Fall back to Docker
                print("Running MMseqs2 via Docker...")
                docker_cmd = ["docker", "run", "--rm", "-v", f"{os.getcwd()}:/app", "-w", "/app"]
                if use_gpu:
                    docker_cmd.extend(["--gpus", "all"])
                docker_cmd.append("ghcr.io/soedinglab/mmseqs2")
                docker_cmd.extend(
                    [
                        "easy-cluster",
                        self._path_in_container(fasta_path),
                        self._path_in_container(output_prefix),
                        self._path_in_container(tmp_dir),
                        "--min-seq-id",
                        str(self.config.cluster_percentage),
                        "-c",
                        str(self.config.coverage),
                        "--threads",
                        str(num_cpu),
                        "--cov-mode",
                        "0",
                        "-s",
                        "7.5",
                        "--split-memory-limit",
                        f"{memory_max}M",
                    ]
                )
                if use_gpu:
                    docker_cmd.extend(["--gpu", "1"])
                self._run_docker_command(docker_cmd, check=True)

        cluster_dict = defaultdict(list)
        with open(cluster_tsv, "r") as f:
            for line in tqdm(f, desc="Reading MMseqs2 cluster file"):
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    rep_id, member_id = parts
                    cluster_dict[rep_id].append(member_id)
        return cluster_dict

    def none(self, seq_dict: Dict[str, str]) -> Dict[str, List[str]]:
        cluster_dict = defaultdict(list)
        for seq_id in seq_dict:
            cluster_dict[seq_id].append(seq_id)
        return cluster_dict

    def __call__(self, seq_dict: Dict[str, str]) -> Dict[str, List[str]]:
        method = self.config.method.lower()
        if method == "cd-hit":
            return self.cdhit(seq_dict)
        if method == "mmseqs2":
            return self.mmseqs2(seq_dict)
        if method == "none":
            return self.none(seq_dict)
        raise ValueError(f"Unknown clustering method: {self.config.method}")


def cluster_sequences(
    seq_dict: Dict[str, str],
    method: str = "mmseqs2",
    cluster_percentage: float = 0.5,
    coverage: float = 0.8,
    n: int = 3,
    memory_percentage: float = 0.5,
    identifier: str = "unknown",
    cache_suffix: str = "",
    base_path: str = "data",
) -> Dict[str, List[str]]:
    config = ClusteringConfig(
        method=method,
        cluster_percentage=cluster_percentage,
        coverage=coverage,
        n=n,
        memory_percentage=memory_percentage,
        cache_suffix=cache_suffix,
        base_path=base_path,
        identifier=identifier,
    )
    return ProteinSequenceClusterer(config)(seq_dict)
