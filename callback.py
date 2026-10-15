def filter_commit(commit, metadata):
    commit.author_name = b"Shashank Rawat"
    commit.author_email = b"rawatshashank1504@gmail.com"
    commit.committer_name = b"Shashank Rawat"
    commit.committer_email = b"rawatshashank1504@gmail.com"
    
    # Clean up attribution text and strip leftover trailing whitespace
    commit.message = (
        commit.message
        .replace(b"Generated with Codebuff", b"")
        .replace(b"Co-Authored-By: Codebuff <noreply@codebuff.com>", b"")
        .strip() + b"\n"
    )