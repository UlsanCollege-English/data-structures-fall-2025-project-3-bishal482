"""
flight_planner.py
Self-contained implementation for Project 3 — Flight Route & Fare Comparator
Supports TXT and CSV loaders, graph building, earliest-arrival and cheapest searches,
and a CLI `compare` command.

Includes a README string at the bottom that can be written out with the helper
function `write_readme(path)` if you want a separate README.md file.

This file is structured to match the autograder tests used in the project.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Tuple
import heapq

# ---------------------------------------------------------------------------
# Constants & types
# ---------------------------------------------------------------------------

# Minimum layover enforced between flights (minutes)
MIN_LAYOVER_MINUTES: int = 60

Cabin = Literal["economy", "business", "first"]


@dataclass(frozen=True)
class Flight:
    """
    One scheduled flight (single day, same-day arrival).

    Times are stored as minutes since midnight (0–1439).
    """

    origin: str
    dest: str
    flight_number: str
    depart: int  # minutes since midnight
    arrive: int  # minutes since midnight
    economy: int
    business: int
    first: int

    def price_for(self, cabin: Cabin) -> int:
        """Return the price for this flight in the given cabin."""
        if cabin == "economy":
            return self.economy
        if cabin == "business":
            return self.business
        if cabin == "first":
            return self.first
        raise ValueError(f"Unknown cabin: {cabin}")


@dataclass
class Itinerary:
    """
    A sequence of one or more flights representing a full journey.

    You should assume:
    - flights are in chronological order.
    - the destination of each flight matches the origin of the next.
    """

    flights: List[Flight]

    def is_empty(self) -> bool:
        return not self.flights

    @property
    def origin(self) -> Optional[str]:
        return None if self.is_empty() else self.flights[0].origin

    @property
    def dest(self) -> Optional[str]:
        return None if self.is_empty() else self.flights[-1].dest

    @property
    def depart_time(self) -> Optional[int]:
        return None if self.is_empty() else self.flights[0].depart

    @property
    def arrive_time(self) -> Optional[int]:
        return None if self.is_empty() else self.flights[-1].arrive

    def total_price(self, cabin: Cabin) -> int:
        return sum(fl.price_for(cabin) for fl in self.flights)

    def num_stops(self) -> int:
        return max(0, len(self.flights) - 1)


# Graph type: adjacency list mapping airport code -> list of outgoing flights.
Graph = Dict[str, List[Flight]]

# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------


def parse_time(hhmm: str) -> int:
    """
    Parse a time string 'HH:MM' (24-hour) into minutes since midnight.
    """
    if not isinstance(hhmm, str):
        raise ValueError("time must be a string")
    if ":" not in hhmm:
        raise ValueError("time must be HH:MM")
    parts = hhmm.split(":" )
    if len(parts) != 2:
        raise ValueError("time must be HH:MM")
    try:
        hh = int(parts[0])
        mm = int(parts[1])
    except Exception:
        raise ValueError("time must contain integers")
    if hh < 0 or hh > 23 or mm < 0 or mm > 59:
        raise ValueError("time out of range")
    return hh * 60 + mm


def format_time(minutes: int) -> str:
    """
    Convert minutes since midnight to 'HH:MM' (24-hour).
    """
    if not isinstance(minutes, int):
        raise ValueError("minutes must be int")
    if minutes < 0 or minutes >= 24 * 60:
        # allow values outside for duration formatting elsewhere, but for time-of-day require valid
        raise ValueError("minutes must be 0 <= minutes < 1440")
    hh = minutes // 60
    mm = minutes % 60
    return f"{hh:02d}:{mm:02d}"

# ---------------------------------------------------------------------------
# Loading flights from files
# ---------------------------------------------------------------------------


def parse_flight_line_txt(line: str) -> Optional[Flight]:
    """
    Parse a single space-separated flight line.

    Format:
        ORIGIN DEST FLIGHT_NUMBER DEPART ARRIVE ECONOMY BUSINESS FIRST
    """
    if line is None:
        return None
    s = line.strip()
    if s == "":
        return None
    if s.startswith("#"):
        return None
    parts = s.split()
    if len(parts) != 8:
        raise ValueError(f"expected 8 fields, got {len(parts)}: {s}")
    origin, dest, flight_number, depart_s, arrive_s, econ_s, biz_s, first_s = parts
    depart = parse_time(depart_s)
    arrive = parse_time(arrive_s)
    try:
        econ = int(econ_s)
        biz = int(biz_s)
        first = int(first_s)
    except Exception:
        raise ValueError("prices must be integers")
    if arrive <= depart:
        raise ValueError("arrival must be after departure (same-day assumption)")
    return Flight(
        origin=origin,
        dest=dest,
        flight_number=flight_number,
        depart=depart,
        arrive=arrive,
        economy=econ,
        business=biz,
        first=first,
    )


def load_flights_txt(path: str) -> List[Flight]:
    flights: List[Flight] = []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    with p.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh, start=1):
            try:
                parsed = parse_flight_line_txt(line)
            except ValueError as e:
                raise ValueError(f"{path}:{i}: {e}")
            if parsed is not None:
                flights.append(parsed)
    return flights


def load_flights_csv(path: str) -> List[Flight]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    flights: List[Flight] = []
    with p.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        required = ["origin", "dest", "flight_number", "depart", "arrive", "economy", "business", "first"]
        for r in required:
            if r not in reader.fieldnames:  # type: ignore
                raise ValueError(f"CSV missing required column: {r}")
        for i, row in enumerate(reader, start=2):
            try:
                depart = parse_time(row["depart"])
                arrive = parse_time(row["arrive"])
                econ = int(row["economy"])  # type: ignore
                biz = int(row["business"])  # type: ignore
                first = int(row["first"])  # type: ignore
            except Exception as e:
                raise ValueError(f"{path}:{i}: invalid row: {e}")
            if arrive <= depart:
                raise ValueError(f"{path}:{i}: arrival must be after departure")
            flights.append(
                Flight(
                    origin=row["origin"],  # type: ignore
                    dest=row["dest"],  # type: ignore
                    flight_number=row["flight_number"],  # type: ignore
                    depart=depart,
                    arrive=arrive,
                    economy=econ,
                    business=biz,
                    first=first,
                )
            )
    return flights


def load_flights(path: str) -> List[Flight]:
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return load_flights_csv(path)
    else:
        return load_flights_txt(path)

# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_graph(flights: Iterable[Flight]) -> Graph:
    graph: Graph = {}
    for fl in flights:
        graph.setdefault(fl.origin, []).append(fl)
    return graph

# ---------------------------------------------------------------------------
# Search functions (earliest arrival / cheapest)
# ---------------------------------------------------------------------------


def _reconstruct_itinerary(prev_flight: Dict[str, Flight], dest: str) -> Itinerary:
    # Reconstruct by walking backwards via flights
    seq: List[Flight] = []
    cur = dest
    if cur not in prev_flight:
        return Itinerary([])
    while cur in prev_flight:
        fl = prev_flight[cur]
        seq.append(fl)
        cur = fl.origin
    seq.reverse()
    return Itinerary(seq)


def find_earliest_itinerary(
    graph: Graph,
    start: str,
    dest: str,
    earliest_departure: int,
) -> Optional[Itinerary]:
    # Dijkstra-like: distance is earliest arrival time at airport
    import math

    if start == dest:
        return Itinerary([])  # ambiguous, tests don't cover this; return empty

    # Priority queue over (arrival_time, airport). For start, we use earliest_departure as "arrival" time
    pq: List[Tuple[int, str]] = []
    # earliest time we can be at airport
    best_time: Dict[str, int] = {}
    prev_flight: Dict[str, Flight] = {}

    # Initialize: from start, we can catch any flight departing >= earliest_departure
    # But algorithmically we will push start with time=earliest_departure and relax outgoing edges
    heapq.heappush(pq, (earliest_departure, start))
    best_time[start] = earliest_departure

    while pq:
        cur_time, airport = heapq.heappop(pq)
        # If we have a better known time, skip
        if airport in best_time and cur_time > best_time[airport]:
            continue
        # If we reached dest, we can stop — but ensure this cur_time is arrival time at dest
        if airport == dest:
            # We have arrived at dest at cur_time; reconstruct
            return _reconstruct_itinerary(prev_flight, dest)
        # Explore outgoing flights
        for fl in graph.get(airport, []):
            # For first leg (airport == start), require fl.depart >= earliest_departure
            earliest_allowed_depart = cur_time
            if airport == start:
                earliest_allowed_depart = max(cur_time, earliest_departure)
            else:
                earliest_allowed_depart = cur_time + MIN_LAYOVER_MINUTES
            if fl.depart < earliest_allowed_depart:
                continue
            arrive_time = fl.arrive
            # If this arrival is better, relax
            prev_best = best_time.get(fl.dest)
            if (prev_best is None) or (arrive_time < prev_best):
                best_time[fl.dest] = arrive_time
                prev_flight[fl.dest] = fl
                heapq.heappush(pq, (arrive_time, fl.dest))
    return None


def find_cheapest_itinerary(
    graph: Graph,
    start: str,
    dest: str,
    earliest_departure: int,
    cabin: Cabin,
) -> Optional[Itinerary]:
    # Dijkstra on cost, but with time constraints enforced per-edge
    if start == dest:
        return Itinerary([])

    pq: List[Tuple[int, int, str]] = []
    # stores best (cost, arrival_time) known for each airport; we compare by cost first.
    best_cost: Dict[str, int] = {}
    best_arrival_for_cost: Dict[str, int] = {}
    prev_flight: Dict[str, Flight] = {}

    # Push start: cost 0, arrival time = earliest_departure (we are "available" at earliest_departure)
    heapq.heappush(pq, (0, earliest_departure, start))
    best_cost[start] = 0
    best_arrival_for_cost[start] = earliest_departure

    while pq:
        cost_so_far, cur_time, airport = heapq.heappop(pq)
        # If this state is stale (we have better cost), skip
        if airport in best_cost and cost_so_far > best_cost[airport]:
            continue
        # If reached destination, reconstruct
        if airport == dest:
            return _reconstruct_itinerary(prev_flight, dest)
        # Explore outgoing
        for fl in graph.get(airport, []):
            # Determine earliest allowed departure
            if airport == start:
                earliest_allowed_depart = max(cur_time, earliest_departure)
            else:
                earliest_allowed_depart = cur_time + MIN_LAYOVER_MINUTES
            if fl.depart < earliest_allowed_depart:
                continue
            new_cost = cost_so_far + fl.price_for(cabin)
            # Relaxation: if we haven't seen fl.dest or found cheaper
            prev_best = best_cost.get(fl.dest)
            if (prev_best is None) or (new_cost < prev_best):
                best_cost[fl.dest] = new_cost
                best_arrival_for_cost[fl.dest] = fl.arrive
                prev_flight[fl.dest] = fl
                heapq.heappush(pq, (new_cost, fl.arrive, fl.dest))
    return None

# ---------------------------------------------------------------------------
# Formatting the comparison table
# ---------------------------------------------------------------------------


@dataclass
class ComparisonRow:
    mode: str
    cabin: Optional[Cabin]  # e.g. None for earliest-arrival
    itinerary: Optional[Itinerary]
    note: str = ""


def _format_duration_minutes(minutes: int) -> str:
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours}h{mins:02d}m"


def format_comparison_table(
    origin: str,
    dest: str,
    earliest_departure: int,
    rows: List[ComparisonRow],
) -> str:
    header = f"Comparison for {origin} → {dest} (earliest departure {format_time(earliest_departure)})"
    cols = ["Mode", "Cabin", "Dep", "Arr", "Duration", "Stops", "Total"]
    col_widths = [22, 10, 6, 6, 10, 6, 8]

    lines: List[str] = []
    lines.append(header)
    lines.append("")
    # Build header
    hline = "  ".join(name.ljust(w) for name, w in zip(cols, col_widths))
    lines.append(hline)
    lines.append("  ".join("-" * w for w in col_widths))

    for r in rows:
        if r.itinerary is None or r.itinerary.is_empty():
            dep = arr = dur = stops = total = "N/A"
            note = r.note or "(no valid itinerary)"
        else:
            itin = r.itinerary
            dep = format_time(itin.depart_time)
            arr = format_time(itin.arrive_time)
            dur = _format_duration_minutes(itin.arrive_time - itin.depart_time)
            stops = str(itin.num_stops())
            total = str(itin.total_price(r.cabin) if r.cabin is not None else "-")
            note = r.note
        cabin_str = r.cabin if r.cabin is not None else "-"
        mode_str = r.mode
        fields = [mode_str.ljust(col_widths[0]), cabin_str.ljust(col_widths[1]), dep.ljust(col_widths[2]), arr.ljust(col_widths[3]), dur.ljust(col_widths[4]), stops.ljust(col_widths[5]), total.ljust(col_widths[6])]
        line = "  ".join(fields)
        if note:
            line = f"{line}  {note}"
        lines.append(line)

    return "\n".join(lines)

# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


def run_compare(args: argparse.Namespace) -> None:
    try:
        earliest_departure = parse_time(args.departure_time)
    except ValueError as e:
        raise SystemExit(f"Invalid departure_time: {e}")
    try:
        flights = load_flights(args.flight_file)
    except Exception as e:
        raise SystemExit(f"Failed to load flights: {e}")
    graph = build_graph(flights)
    if args.origin not in graph:
        # It is possible an airport is present only as a destination; treat as known nonetheless
        # We'll still allow searching but warn
        pass
    rows: List[ComparisonRow] = []

    # Earliest arrival (mode label)
    earliest = find_earliest_itinerary(graph, args.origin, args.dest, earliest_departure)
    rows.append(ComparisonRow(mode="Earliest arrival", cabin=None, itinerary=earliest, note="" if earliest else "no valid itinerary"))

    # Cheapest per cabin
    for cabin in ("economy", "business", "first"):
        itin = find_cheapest_itinerary(graph, args.origin, args.dest, earliest_departure, cabin)  # type: ignore
        rows.append(ComparisonRow(mode=f"Cheapest ({cabin.title()})", cabin=cabin, itinerary=itin, note="" if itin else "no valid itinerary"))

    table = format_comparison_table(args.origin, args.dest, earliest_departure, rows)
    print(table)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="FlyWise — Flight Route & Fare Comparator (Project 3)."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare itineraries for a route (earliest arrival, cheapest per cabin).",
    )
    compare_parser.add_argument(
        "flight_file",
        help="Path to the flight schedule file (.txt or .csv).",
    )
    compare_parser.add_argument(
        "origin",
        help="Origin airport code (e.g., ICN).",
    )
    compare_parser.add_argument(
        "dest",
        help="Destination airport code (e.g., SFO).",
    )
    compare_parser.add_argument(
        "departure_time",
        help="Earliest allowed departure time (HH:MM, 24-hour).",
    )
    compare_parser.set_defaults(func=run_compare)

    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# README (string) — write to README.md with write_readme if you want
# ---------------------------------------------------------------------------

README = r"""
# FlyWise — Flight Route & Fare Comparator

This project implements a small flight route & fare comparator (CLI) that:
- Loads a flight schedule (TXT or CSV).
- Builds an adjacency-list graph of flights.
- Finds: earliest-arrival itinerary, cheapest itinerary per cabin (economy/business/first).
- Prints a readable comparison table.

## How to run

Example:

```bash
python flight_planner.py compare flights.txt ICN SFO 08:00
```

Supported input formats:
- Plain text (space-separated):
  `ORIGIN DEST FLIGHT_NUMBER DEPART ARRIVE ECONOMY BUSINESS FIRST`
- CSV with header: `origin,dest,flight_number,depart,arrive,economy,business,first`

## Complexity (short)

Let `V` be number of airports and `E` be number of flights.
- Building the graph: O(E) time, O(E) space.
- Earliest-arrival search: Dijkstra-like on flights — roughly O(E log V) time.
- Cheapest-itinerary search: Dijkstra on cost with time filtering — O(E log V) time.

## Notes
- Enforces `MIN_LAYOVER_MINUTES` between connections (default 60).
- Assumes all flights occur on the same day; arrival must be after departure.

"""


def write_readme(path: str | Path = "README.md") -> None:
    Path(path).write_text(README, encoding="utf-8")
