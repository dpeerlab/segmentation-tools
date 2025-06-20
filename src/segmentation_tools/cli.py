"""Console script for segmentation_tools."""
import segmentation_tools

import typer
from rich.console import Console

app = typer.Typer()
console = Console()


@app.command()
def main():
    """Console script for segmentation_tools."""
    console.print("Replace this message by putting your code into "
               "segmentation_tools.cli.main")
    console.print("See Typer documentation at https://typer.tiangolo.com/")
    


if __name__ == "__main__":
    app()
