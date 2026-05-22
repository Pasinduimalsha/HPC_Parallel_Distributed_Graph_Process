#!/usr/bin/env python3
import argparse
import os
import struct
from array import array
from pathlib import Path


HEADER = struct.Struct("<8si4xqqii")
MAGIC = b"HPCPRCSR"
VERSION = 1


def read_edges(graph_path: Path):
    src = array("i")
    dst = array("i")
    max_vertex = -1

    with graph_path.open("r") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                u = int(parts[0])
                v = int(parts[1])
            except ValueError:
                continue
            if u < 0 or v < 0:
                continue

            src.append(u)
            dst.append(v)
            if u > max_vertex:
                max_vertex = u
            if v > max_vertex:
                max_vertex = v

    return src, dst, max_vertex + 1


def prefix_index(degree):
    index = array("i", [0]) * (len(degree) + 1)
    total = 0
    for i, count in enumerate(degree):
        index[i] = total
        total += count
    index[len(degree)] = total
    return index


def write_array(handle, values):
    values.tofile(handle)


def convert(graph_path: Path, output_path: Path, overwrite: bool) -> tuple[int, int]:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists; use --overwrite")

    src, dst, vertices = read_edges(graph_path)
    edges = len(src)

    out_degree = array("i", [0]) * vertices
    in_degree = array("i", [0]) * vertices
    for u, v in zip(src, dst):
        out_degree[u] += 1
        in_degree[v] += 1

    adjacency_index = prefix_index(out_degree)
    in_adjacency_index = prefix_index(in_degree)
    adjacency_list = array("i", [0]) * edges
    in_adjacency_list = array("i", [0]) * edges

    out_cursor = array("i", adjacency_index[:-1])
    in_cursor = array("i", in_adjacency_index[:-1])
    for u, v in zip(src, dst):
        pos = out_cursor[u]
        adjacency_list[pos] = v
        out_cursor[u] += 1

        in_pos = in_cursor[v]
        in_adjacency_list[in_pos] = u
        in_cursor[v] += 1

    stat = graph_path.stat()
    header = HEADER.pack(
        MAGIC,
        VERSION,
        int(stat.st_size),
        int(stat.st_mtime),
        vertices,
        edges,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        f.write(header)
        write_array(f, out_degree)
        write_array(f, adjacency_index)
        write_array(f, adjacency_list)
        write_array(f, in_degree)
        write_array(f, in_adjacency_index)
        write_array(f, in_adjacency_list)

    return vertices, edges


def main():
    parser = argparse.ArgumentParser(
        description="Convert edge-list graph .txt files into HPC PageRank .csr cache files."
    )
    parser.add_argument("graph", nargs="+", type=Path, help="edge-list .txt graph file(s)")
    parser.add_argument(
        "--overwrite", action="store_true", help="replace existing .csr files"
    )
    args = parser.parse_args()

    if array("i").itemsize != 4:
        raise RuntimeError("This platform does not use 4-byte C int arrays.")

    for graph_path in args.graph:
        if graph_path.suffix != ".txt":
            raise ValueError(f"Expected a .txt graph file: {graph_path}")
        output_path = graph_path.with_suffix(".csr")
        vertices, edges = convert(graph_path, output_path, args.overwrite)
        print(f"created {output_path}: {vertices} vertices, {edges} edges")


if __name__ == "__main__":
    main()
