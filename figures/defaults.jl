using Plots, StatsPlots

gr()
default()
default(
    fontfamily="Helvetica",
    tick_direction=:out,
    guidefontsize=9,
    annotationfontfamily="Helvetica",
    annotationfontsize=10,
    annotationhalign=:left,
    box=:on,
    msw=0.0,
    lw=1.5
)

alphabet = "abcdefghijklmnopqrstuvwxyz"

function add_plot_labels!(plt;offset=0)
    n = length(plt.subplots)
    for i = 1:n
        plot!(plt,subplot=i,title="($(alphabet[i+offset]))")
    end
    plot!(
        titlelocation = :left,
        titlefontsize = 10,
        titlefontfamily = "Helvetica"
    )
end

# Scale diverging colormap based on 0
using ColorSchemes
function centered_cmap(γ,cmap::ColorPalette=reverse(palette(:bwr)))
    func(x) = x < γ ? x / (2γ) : 0.5 + (x - γ) / (2*(1 - γ))
    x = range(0.0,1.0,101)
    palette(ColorScheme([get(cmap,func(xᵢ)) for xᵢ in x]))
end
centered_cmap(a::Number,b::Number,cmap::ColorPalette=reverse(palette(:bwr))) = centered_cmap(-a / (b - a),cmap)
centered_cmap(cmap::ColorPalette=reverse(palette(:bwr))) = centered_cmap(zlims(plot!())...,cmap)


function clipped_cmap(l,u=1.0;cmap=palette(:Blues))
    palette(ColorScheme([get(cmap,l + (u - l) * x) for x in range(0.0,1.0,101)]))
end