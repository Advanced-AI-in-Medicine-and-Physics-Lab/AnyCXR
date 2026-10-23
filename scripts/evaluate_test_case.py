"""Compatibility entry point; use scripts/evaluate.py for evaluation."""
if __package__:
    from .evaluate import main
else:
    from evaluate import main

if __name__ == "__main__":
    main()
