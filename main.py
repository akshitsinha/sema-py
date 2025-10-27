import click
from pathlib import Path
from rich.console import Console
from rich.progress import track
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from text import SemanticSearchEngine


console = Console()


@click.group()
def cli():
    """Semantic search tool for local text files."""
    pass


@cli.command()
@click.option('--input-dir', required=True, type=click.Path(exists=True), help='Directory containing text files')
@click.option('--extensions', multiple=True, default=['.txt'], help='File extensions to index (can specify multiple)')
@click.option('--force', is_flag=True, help='Force re-index all files')
@click.option('--chunk-size', default=800, help='Size of text chunks in characters')
@click.option('--chunk-overlap', default=100, help='Overlap between chunks in characters')
def index(input_dir, extensions, force, chunk_size, chunk_overlap):
    """Index files in the specified directory."""
    console.print(f"\n[bold cyan]Indexing files from:[/bold cyan] {input_dir}")
    console.print(f"[dim]Extensions: {', '.join(extensions)}[/dim]")
    console.print(f"[dim]Chunk size: {chunk_size} chars, Overlap: {chunk_overlap} chars[/dim]\n")
    
    engine = SemanticSearchEngine(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    with console.status("[bold green]Scanning files..."):
        stats = engine.index_directory(
            input_dir=input_dir,
            file_extensions=list(extensions),
            force=force
        )
    
    console.print(f"[bold green]✓[/bold green] Indexed {stats['new']} new files")
    console.print(f"[bold yellow]⟳[/bold yellow] Updated {stats['updated']} changed files")
    console.print(f"[dim]⊘ Skipped {stats['skipped']} unchanged files[/dim]")
    console.print(f"\n[bold]Total:[/bold] {stats['total']} files → {stats['chunks']} chunks\n")


@cli.command()
@click.argument('query')
@click.option('--top', default=5, help='Number of results to show')
@click.option('--context', default=5, help='Number of context lines around match')
def search(query, top, context):
    """Search for semantic matches."""
    engine = SemanticSearchEngine()
    
    console.print(f"\n[bold cyan]🔍 Searching for:[/bold cyan] [yellow]{query}[/yellow]\n")
    
    with console.status("[bold green]Searching..."):
        results = engine.search(query, n_results=top)
    
    if not results:
        console.print("[red]No results found.[/red]\n")
        return
    
    console.print(f"[bold green]Found {len(results)} results:[/bold green]\n")
    console.print("━" * console.width)
    
    for i, result in enumerate(results, 1):
        file_path = result['file_path']
        score = result['score']
        line_start = result['line_start']
        line_end = result['line_end']
        
        file_name = Path(file_path).name
        
        context_text, ctx_start, ctx_end = engine.get_context(
            file_path, line_start, line_end, context
        )
        
        title = f"📄 {file_name} [dim](Score: {score:.2f})[/dim]"
        subtitle = f"Lines {line_start}-{line_end}"
        
        lines = context_text.split('\n')
        formatted_lines = []
        
        for line_num, line in enumerate(lines, start=ctx_start):
            prefix = f"[dim]{line_num:4d} |[/dim] "
            
            # Highlight lines that are in the matched range
            if line_start <= line_num <= line_end:
                formatted_lines.append(prefix + f"[bold yellow]{line}[/bold yellow]")
            else:
                formatted_lines.append(prefix + f"[dim]{line}[/dim]")
        
        content = '\n'.join(formatted_lines)
        
        panel = Panel(
            content,
            title=title,
            subtitle=subtitle,
            border_style="cyan",
            expand=False
        )
        
        console.print(panel)
        console.print()


@cli.command()
@click.confirmation_option(prompt='Are you sure you want to clear all indexed data?')
def clear():
    """Clear all indexed data."""
    engine = SemanticSearchEngine()
    
    with console.status("[bold red]Clearing database..."):
        engine.clear()
    
    console.print("[bold green]✓[/bold green] Database cleared.\n")


@cli.command()
def status():
    """Show database statistics."""
    engine = SemanticSearchEngine()
    
    with console.status("[bold green]Gathering stats..."):
        stats = engine.get_stats()
    
    table = Table(title="Database Statistics", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Total Files", str(stats['total_files']))
    table.add_row("Total Chunks", str(stats['total_chunks']))
    table.add_row("Database Size", f"{stats['total_size_mb']} MB")
    
    console.print()
    console.print(table)
    console.print()


if __name__ == '__main__':
    cli()
