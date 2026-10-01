# Outils de test manuel (hors livrables officiels)

Ce dossier n'est **pas** un des 6 micro-services du projet. Il sert uniquement à
tester l'`api-gateway` (Service 5) de bout en bout, en local, avant que les
Services 3 (`audio-engine`) et 4 (`agent-orchestrator`) ne soient développés.

## Démarrage (3 terminaux)

```bash
# Terminal 1 : stub du Service 3
uvicorn tools.stubs.stub_audio_engine:app --port 8003

# Terminal 2 : stub du Service 4
uvicorn tools.stubs.stub_agent_orchestrator:app --port 8004

# Terminal 3 : l'api-gateway réel, pointé vers les stubs
AUDIO_ENGINE_URL=http://localhost:8003 AGENT_ORCHESTRATOR_URL=http://localhost:8004 \
  uvicorn api.main:app --port 8000
```

Sous PowerShell :
```powershell
$env:AUDIO_ENGINE_URL="http://localhost:8003"
$env:AGENT_ORCHESTRATOR_URL="http://localhost:8004"
uvicorn api.main:app --port 8000
```

Puis ouvrez [tools/web-tester/index.html](web-tester/index.html) dans un navigateur
(double-clic suffit), choisissez un fichier `.wav` et cliquez sur "Envoyer".

## Ce que ça valide

- La chaîne complète api-gateway → audio-engine → agent-orchestrator → client.
- La validation d'entrée (essayez un fichier non-`.wav` pour voir le 422).
- Les erreurs 502/504 : arrêtez un des deux stubs et renvoyez une requête.

## Remplacer les stubs

Une fois les vrais Services 3 et 4 prêts, ces stubs ne servent plus à rien :
pointez simplement `AUDIO_ENGINE_URL`/`AGENT_ORCHESTRATOR_URL` vers les vrais
services (ou laissez les valeurs par défaut de `.env` une fois dans
`docker-compose.yml`).
