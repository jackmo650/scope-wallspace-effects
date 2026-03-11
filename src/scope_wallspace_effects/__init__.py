from scope.core.plugins.hookspecs import hookimpl


@hookimpl
def register_pipelines(register):
    from .pipeline import WallspaceEffectsPrePipeline, WallspaceEffectsPostPipeline

    register(WallspaceEffectsPrePipeline)
    register(WallspaceEffectsPostPipeline)
