import warnings


def main():
    warnings.simplefilter("always", DeprecationWarning)

    warnings.warn(
        "The CLI and templating using evalutils is deprecated. "
        "Use the templates provided by the challenge organisers instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    raise (SystemExit(1))


if __name__ == "__main__":
    main()
