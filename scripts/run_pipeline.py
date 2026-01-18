import argparse

from hmsynth.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", default="configs", help="Directory containing schema/motif/optimization/generation yaml")
    parser.add_argument("--survey_path", default=None, help="Override survey CSV path")
    parser.add_argument("--output_dir", default="outputs", help="Output directory for motifs/margins/best_x")
    args = parser.parse_args()

    run_pipeline(config_dir=args.config_dir, survey_path=args.survey_path, output_dir=args.output_dir)


if __name__ == "__main__":
    main()

