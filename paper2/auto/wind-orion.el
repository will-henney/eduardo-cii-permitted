(TeX-add-style-hook
 "wind-orion"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("mnras" "useAMS" "usenatbib")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("babel" "spanish" "es-minimal" "english") ("inputenc" "utf8") ("newtxmath" "stix2" "smallerops") ("enumitem" "shortlabels")))
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "href")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperref")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperimage")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperbaseurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "nolinkurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (TeX-run-style-hooks
    "latex2e"
    "mnras"
    "mnras10"
    "babel"
    "inputenc"
    "graphicx"
    "xcolor"
    "hyperref"
    "siunitx"
    "newtxtext"
    "newtxmath"
    "booktabs"
    "etoolbox"
    "enumitem")
   (TeX-add-symbols
    '("chem" 1)
    '("Wav" 1)
    '("ION" 2)
    "hii"
    "nii"
    "oiii"
    "oii"
    "ha"
    "Fion"
    "ionpar")
   (LaTeX-add-labels
    "firstpage"
    "sec:introduction"
    "fig:v-hist-quadrant"
    "sec:muse-observ-orion"
    "sec:discussion"
    "sec:nature-big-arc"
    "sec:conclusions"
    "lastpage")
   (LaTeX-add-bibliographies
    "wind-orion-refs")
   (LaTeX-add-counters
    "ionstage")
   (LaTeX-add-siunitx-units
    "msun"
    "lsun"))
 :latex)

