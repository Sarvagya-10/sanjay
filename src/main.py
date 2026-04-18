from ingest import ingest
from rank import rank_papers
from summarize import summarize_papers
from telegram import deliver


def main():
    papers = ingest()
    print("INGEST:", len(papers))

    top = rank_papers(papers)
    print("RANKED:", len(top))

    summaries = summarize_papers(top)
    print("SUMMARIZED:", len(summaries))

    deliver(summaries)


if __name__ == "__main__":
    main()