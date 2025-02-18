def calculate_monthly_compound_interest(principal: float, yearly_rate: float, months: int, monthly_expense: float) -> float:
    """
    Calculate compound interest with monthly expenses.
    
    Args:
        principal: Initial investment amount
        yearly_rate: Annual interest rate as a percentage
        months: Number of months to calculate for
        monthly_expense: Monthly withdrawal/expense amount
    
    Returns:
        Final balance after compound interest and expenses
    """
    monthly_rate = yearly_rate / 12 / 100  # Convert yearly rate to monthly decimal
    balance = principal
    
    # Apply monthly compounding and subtract expenses for each month
    for _ in range(months):
        # Add monthly interest
        balance = balance * (1 + monthly_rate)
        # Subtract monthly expense
        balance = balance - monthly_expense
    
    return max(0, balance)  # Prevent negative balance

def generate_schedule(principal: float, yearly_rate: float, years: int, monthly_expense: float) -> list:
    """
    Generate a yearly schedule of balances.
    
    Args:
        principal: Initial investment amount
        yearly_rate: Annual interest rate as a percentage
        years: Number of years to calculate for
        monthly_expense: Monthly withdrawal/expense amount
    
    Returns:
        List of [year, balance] pairs
    """
    schedule = []
    
    for year in range(years + 1):
        months = year * 12
        balance = calculate_monthly_compound_interest(principal, yearly_rate, months, monthly_expense)
        schedule.append([year, balance])
    
    return schedule

def format_currency(amount: float) -> str:
    """Format a number as US currency string."""
    return f"${amount:,.2f}"

def calculate(principal: float, rate: float, time: int, monthly_expense: float) -> tuple[str, str]:
    """
    Calculate compound interest and generate formatted results.
    
    Args:
        principal: Initial investment amount
        rate: Annual interest rate as a percentage
        time: Time period in years
        monthly_expense: Monthly withdrawal/expense amount
    
    Returns:
        Tuple of (results_string, schedule_string)
    """
    final_amount = calculate_monthly_compound_interest(principal, rate, time * 12, monthly_expense)
    
    # Format results
    results = f"""
Results:
Initial Investment: {format_currency(principal)}
Interest Rate: {rate}%
Time Period: {time} years
Monthly Expense: {format_currency(monthly_expense)}
Final Amount: {format_currency(final_amount)}
"""
    
    # Generate and format schedule
    schedule = generate_schedule(principal, rate, time, monthly_expense)
    schedule_lines = ["Yearly Investment Schedule:", "Year    Balance"]
    
    for year, balance in schedule:
        schedule_lines.append(f"Year {year:<4} {format_currency(balance)}")
    
    schedule_text = "\n".join(schedule_lines)
    
    return results.strip(), schedule_text

def main():
    """Example usage of the compound interest calculator."""
    # Example values
    principal = 100000
    rate = 5.0
    time = 10
    monthly_expense = 500
    
    results, schedule = calculate(principal, rate, time, monthly_expense)
    print(results)
    print("\n" + schedule)

if __name__ == "__main__":
    main()
