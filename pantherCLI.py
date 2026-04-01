import sys
import argparse
from pathlib import Path
import torch
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

# Importiamo le funzioni dal tuo file model.py esistente
try:
    from model import load_model, predict_top_k
except ImportError:
    # Se il file si chiama diversamente o le funzioni sono altrove, 
    # assicurati che siano accessibili
    print("Errore: Assicurati che 'model.py' sia nella stessa cartella.")
    sys.exit(1)

console = Console()

def parse_fasta(text: str) -> str:
    """Pulisce l'input rimuovendo header FASTA e spazi."""
    lines = text.strip().splitlines()
    if not lines:
        return ""
    seq_lines = [l.strip() for l in lines if not l.startswith(">")]
    return "".join(seq_lines).replace(" ", "").upper()

def main():
    parser = argparse.ArgumentParser(description="PANTHER Protein Classifier CLI")
    parser.add_argument("sequence", nargs="?", help="La sequenza proteica da analizzare")
    parser.add_argument("--weights", default="best_panther_model_pro.pth", help="Percorso file .pth")
    parser.add_argument("--classes", default="classes_pro.npy", help="Percorso file .npy")
    parser.add_argument("--topk", type=int, default=5, help="Numero di risultati da mostrare")
    args = parser.parse_args()

    # Se non viene passata la sequenza come argomento, la chiediamo in input
    if not args.sequence:
        console.print("[bold cyan] PANTHER Classifier CLI v6[/bold cyan]")
        args.sequence = console.input("[bold]Incolla la sequenza proteica e premi Invio: [/bold]\n")

    seq = parse_fasta(args.sequence)
    if not seq:
        console.print("[bold red] Sequenza non valida o vuota![/bold red]")
        return

    # Caricamento Modello
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            progress.add_task(description="Caricamento modello...", total=None)
            model, label_classes = load_model(args.weights, args.classes)
            
            progress.add_task(description="Elaborazione predizione...", total=None)
            results = predict_top_k(model, label_classes, seq, k=args.topk)
    except Exception as e:
        console.print(f"[bold red]❌ Errore durante il caricamento o la predizione:[/bold red] {e}")
        return

    # Visualizzazione Risultati
    console.print(f"\n[bold green]Analisi completata![/bold green] Lunghezza sequenza: [yellow]{len(seq)} aa[/yellow]")
    
    table = Table(title="Top Predizioni PANTHER", title_style="bold magenta")
    table.add_column("Rank", justify="right", style="dim")
    table.add_column("Famiglia", style="cyan", no_wrap=True)
    table.add_column("Confidenza", justify="right", style="green")
    table.add_column("Barra", width=20)

    for i, (name, prob) in enumerate(results, start=1):
        bar_size = int(prob / 5)  # 20 caratteri max (100/5)
        bar = "|" * bar_size
        table.add_row(
            str(i),
            name,
            f"{prob:.2f}%",
            f"[green]{bar}[/green]"
        )

    console.print(table)
    
    # Mostriamo la famiglia vincitrice in un pannello dedicato
    top_name, top_prob = results[0]
    console.print(Panel(
        f"La proteina appartiene molto probabilmente alla famiglia:\n[bold cyan]{top_name}[/bold cyan] con il [bold green]{top_prob:.2f}%[/bold green] di confidenza.",
        title="Risultato Principale",
        border_style="green"
    ))

if __name__ == "__main__":
    main()
