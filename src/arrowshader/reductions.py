from dataclasses import dataclass


@dataclass(frozen=True)
class Count:
    pass


@dataclass(frozen=True)
class Sum:
    column: str


@dataclass(frozen=True)
class Mean:
    column: str


@dataclass(frozen=True)
class By:
    column: str


@dataclass(frozen=True)
class Min:
    column: str


@dataclass(frozen=True)
class Max:
    column: str


@dataclass(frozen=True)
class Std:
    column: str


@dataclass(frozen=True)
class Var:
    column: str


@dataclass(frozen=True)
class First:
    column: str


@dataclass(frozen=True)
class Last:
    column: str


@dataclass(frozen=True)
class Any:
    column: str


def count() -> Count:
    return Count()


def sum(column: str) -> Sum:
    return Sum(column)


def mean(column: str) -> Mean:
    return Mean(column)


def by(column: str) -> By:
    return By(column)


def min(column: str) -> Min:
    return Min(column)


def max(column: str) -> Max:
    return Max(column)


def std(column: str) -> Std:
    return Std(column)


def var(column: str) -> Var:
    return Var(column)


def first(column: str) -> First:
    return First(column)


def last(column: str) -> Last:
    return Last(column)


def any(column: str) -> Any:
    return Any(column)
