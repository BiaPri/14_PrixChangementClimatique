# Architecture

Le contenu de ce dossier permet de générer les données utilisées par le rendu final.

Ces données sont générées grâce à un projet DBT.

# DBT

## Documentation

Pour visualiser le contenu du projet dbt (table et process), lancer les commandes suivantes depuis le dossier dbt (`cd data/dbt`)

```bash
uv run dbt docs generate
uv run dbt docs serve
```

L'interface est alors disponible sur votre http://localhost:8080/

# WIP - Liste des data utilisées / disponibles

## Données assurance

(dossier /Gaspar) Base GASPAR publiée sur DataGouv : https://www.data.gouv.fr/datasets/base-nationale-de-gestion-assistee-des-procedures-administratives-relatives-aux-risques-gaspar/

(TODO) Arrêtés Cat Nat (dont cat nat non reconnues) : https://www.ccr.fr/portail-catastrophes-naturelles/liste-arretes/

## Données socio/demo

(submodule /13_odis) Repo du projet Odis saison 13 de D4G, données collectées listées dans le fichier datasources.yaml
