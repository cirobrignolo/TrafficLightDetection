# Tesis - Sistema de Detección y Reconocimiento de Semáforos

Este directorio contiene el código LaTeX para el informe final de la tesis.

## Estructura

```
latex/
├── tesis.tex              # Documento principal
├── caratula.sty           # Estilo de carátula (UBA)
├── chapters/              # Capítulos de la tesis
│   ├── 01_introduccion.tex
│   ├── 02_marco_teorico.tex
│   ├── 03_arquitectura_apollo.tex
│   ├── 04_implementacion.tex
│   ├── 05_resultados.tex
│   ├── 06_conclusiones.tex
│   └── apendice_a.tex
├── figures/               # Imágenes y figuras (agregar aquí)
├── bibliography/          # Archivos de bibliografía
│   └── referencias.bib
├── logodc.jpg            # Logo Departamento de Computación
└── logouba.jpg           # Logo UBA
```

## Uso en Overleaf

1. Crear un nuevo proyecto en [Overleaf](https://www.overleaf.com)
2. Subir todos los archivos de este directorio
3. Configurar el compilador principal como `tesis.tex`
4. Compilar

## TODOs

Los capítulos contienen marcadores `TODO` que indican secciones que necesitan ser completadas:

- Expandir secciones teóricas con ecuaciones y explicaciones detalladas
- Agregar diagramas de arquitectura
- Completar resultados experimentales con datos reales
- Agregar figuras y gráficos en el directorio `figures/`
- Expandir la bibliografía en `bibliography/referencias.bib`

## Personalización

### Información de la carátula

Editar en `tesis.tex` las líneas de información personal:

```latex
\materia{Licenciatura en Ciencias de la Computación}
\titulo{Sistema de Detección y Reconocimiento de Semáforos}
\integrante{Tu Nombre}{LU}{email@example.com}
```

### Agregar figuras

1. Subir las imágenes al directorio `figures/` en Overleaf
2. Incluir en el capítulo correspondiente:

```latex
\begin{figure}[H]
\centering
\includegraphics[width=0.8\textwidth]{figures/mi_imagen.png}
\caption{Descripción de la imagen}
\label{fig:mi_imagen}
\end{figure}
```

### Agregar referencias bibliográficas

Editar `bibliography/referencias.bib` y agregar entradas BibTeX, luego citar con `\cite{clave}`
