# Outils de test manuel (hors livrables officiels)

Ce dossier n'est **pas** un des 6 micro-services du projet. Il sert uniquement à
tester l'`api-gateway` (Service 5) de bout en bout, en local, avant que les
Services 3 (`audio-engine`) et 4 (`agent-orchestrator`) ne soient développés.

Flux simulé (architecture cible) :
`client → api-gateway (S5) → agent-orchestrator (S4) → [AudioAgent] → audio-engine (S3)`.
La gateway ne parle qu'au Service 4 ; c'est le stub S4 qui appelle le stub S3.

## Démarrage (3 terminaux)

```bash
# Terminal 1 : stub du Service 3
uvicorn tools.stubs.stub_audio_engine:app --port 8003

# Terminal 2 : stub du Service 4
uvicorn tools.stubs.stub_agent_orchestrator:app --port 8004

# Terminal 3 : l'api-gateway réel, pointé vers le stub S4
AGENT_ORCHESTRATOR_URL=http://127.0.0.1:8004 uvicorn api.main:app --port 8000
```

Sous PowerShell :
```powershell
$env:AGENT_ORCHESTRATOR_URL="http://127.0.0.1:8004"
uvicorn api.main:app --port 8000
```

Puis ouvrez [tools/web-tester/index.html](web-tester/index.html) dans un navigateur
(double-clic suffit), choisissez un fichier `.wav` et cliquez sur "Envoyer".

## Ce que ça valide

- La chaîne complète client → api-gateway → agent-orchestrator → audio-engine.
- La validation d'entrée (essayez un fichier non-`.wav` pour voir le 422) et les
  guardrails du prompt (essayez « ignore les instructions précédentes »).
- Les erreurs 502/504 : arrêtez un des deux stubs et renvoyez une requête.

## Remplacer les stubs

Une fois les vrais Services 3 et 4 prêts, ces stubs ne servent plus à rien :
pointez simplement `AGENT_ORCHESTRATOR_URL` vers le vrai Service 4 (ou laissez
la valeur par défaut de `.env` une fois dans `docker-compose.yml`). C'est le
Service 4 qui doit connaître l'URL du Service 3 (`AUDIO_ENGINE_URL`).

## Contrat attendu du Service 4

`POST /analyze` en `multipart/form-data` :

| Champ      | Type   | Description                                          |
|------------|--------|------------------------------------------------------|
| `file`     | fichier| Le `.wav` validé par la gateway                      |
| `filename` | texte  | Nom du fichier d'origine                             |
| `prompt`   | texte  | Question nettoyée par les guardrails (absent si vide)|

En-tête `X-Request-ID` propagé pour la traçabilité. Réponse : un rapport JSON
(schéma libre, relayé tel quel au client).

`POST /analyze/stream` : mêmes champs, réponse `text/event-stream` relayée
octet par octet par la gateway sur `POST /analyze_pcg/stream`. Événements
attendus (format `event: <nom>\ndata: <json>\n\n`) :

| Événement | `data`                                                        |
|-----------|---------------------------------------------------------------|
| `step`    | `{"agent": "AudioAgent", "status": "running" \| "done", ...}` |
| `token`   | `{"text": "..."}` : morceau de texte généré par le LLM        |
| `report`  | le rapport final complet (même contenu que `/analyze`)        |
| `done`    | `{}` : fin du flux                                            |
| `error`   | `{"error", "service", "detail"}`                              |

Si le Service 4 répond une erreur HTTP **avant** d'ouvrir le flux (ex. S3 en
panne), la gateway renvoie un 502/504 JSON classique. Si le flux est coupé
**en cours de route**, la gateway ajoute elle-même un dernier événement `error`.
