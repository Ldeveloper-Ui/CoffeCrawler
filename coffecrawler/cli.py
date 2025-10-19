"""
☕ CoffeCrawler CLI - Command Line Interface
"""

import click
from typing import Optional

@click.group()
@click.version_option()
def main():
    """☕ CoffeCrawler - Next Generation AI-Powered Web Crawling Library"""
    pass

@main.command()
@click.argument('url')
@click.option('--mode', '-m', default='smart', 
              type=click.Choice(['smart', 'stealth', 'aggressive', 'safe']),
              help='Crawling mode')
@click.option('--output', '-o', type=click.Path(), help='Output file')
def crawl(url: str, mode: str, output: Optional[str]):
    """Crawl a website with AI-powered strategies"""
    from coffecrawler import CoffeCrawler
    
    crawler = CoffeCrawler(mode=mode)
    result = crawler.crawl(url)
    
    if output:
        import json
        with open(output, 'w') as f:
            json.dump(result, f, indent=2)
        click.echo(f"✅ Results saved to: {output}")
    else:
        click.echo(json.dumps(result, indent=2))

@main.command()
def strategies():
    """List all available AI strategies"""
    from coffecrawler import get_strategy_analytics
    analytics = get_strategy_analytics()
    click.echo("🎯 Available Strategies:")
    for strategy in analytics.get('top_performing_strategies', []):
        click.echo(f"  • {strategy['strategy']}: {strategy['success_rate']:.1%} success rate")

@main.command()
def dashboard():
    """Show system dashboard"""
    from coffecrawler import system_dashboard
    import json
    dashboard = system_dashboard()
    click.echo("📊 CoffeCrawler System Dashboard:")
    click.echo(json.dumps(dashboard, indent=2))

if __name__ == '__main__':
    main()
