class: CommandLineTool
cwlVersion: v1.2
inputs:
  cropX:
    inputBinding:
      prefix: --cropX
    type: boolean?
  cropY:
    inputBinding:
      prefix: --cropY
    type: boolean?
  cropIndividually:
    inputBinding:
      prefix: --cropIndividually
    type: boolean?
  filePattern:
    inputBinding:
      prefix: --filePattern
    type: string
  groupBy:
    inputBinding:
      prefix: --groupBy
    type: string
  inpDir:
    inputBinding:
      prefix: --inpDir
    type: Directory
  outDir:
    inputBinding:
      prefix: --outDir
    type: Directory
outputs:
  outDir:
    outputBinding:
      glob: $(inputs.outDir.basename)
    type: Directory
requirements:
  DockerRequirement:
    dockerPull: polusai/autocropping-plugin:2.0.0
