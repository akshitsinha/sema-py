import click
from repl import SearchREPL


@click.command()
@click.argument('directory', type=click.Path(exists=True))
@click.option('--extensions', multiple=True, default=['.txt', '.md', '.pdf'], help='File extensions to index')
@click.option('--chunk-size', default=800, help='Chunk size in characters')
@click.option('--chunk-overlap', default=100, help='Chunk overlap in characters')
def main(directory, extensions, chunk_size, chunk_overlap):
    """Start semantic search REPL for DIRECTORY."""
    repl = SearchREPL(
        directory=directory,
        extensions=list(extensions),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    repl.run()


if __name__ == '__main__':
    main()
